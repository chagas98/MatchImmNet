import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, global_mean_pool, global_add_pool, GraphNorm
from torch_geometric.utils import to_dense_batch
from pyGeoMatchImm.utils.base import TrainConfigs
from ..embeddings.positional import PositTCREncoder, PositEpitopeEncoder, PositMHCEncoder

torch.manual_seed(42); np.random.seed(42)

# -----------------------------
# 1. Cross-Attention module
# -----------------------------
class CrossAttention(nn.Module):
    """
    Cross-attention: queries from one encoder attend to keys/values from the other.
    Used bidirectionally (A->B and B->A) inside the model.
    """
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv):
        Bq, D = x_q.shape
        Bk, _ = x_kv.shape
        H = self.heads

        # Linear projections
        Q = self.q_proj(x_q).view(Bq, H, int(D // H))
        K = self.k_proj(x_kv).view(Bk, H, int(D // H))
        V = self.v_proj(x_kv).view(Bk, H, int(D // H))

        # Attention weights
        attn = torch.einsum('qhd,khd->hqk', Q, K) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Contextualized output
        out = torch.einsum('hqk,khd->qhd', attn, V).reshape(Bq, D)
        out = self.out_proj(out)
        return out

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.cross = CrossAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.LeakyReLU(),
            nn.Linear(dim*2, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x_q, x_kv):
        attn_out = self.cross(x_q, x_kv)
        x = self.norm1(x_q + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class CrossAttentionNodesBlockv1(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.cross = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.LeakyReLU(),
            nn.Linear(dim*2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, Q, K, V, key_padding_mask=None, attention_mask=None, query_padding_mask=None):
        # attention_mask: [B, Lq, Lk] (bool, True = mask out)
        attn_mask_3d = None
        if attention_mask is not None:
            attn_mask = attention_mask.to(torch.bool)

            if query_padding_mask is not None:
                attn_mask = attn_mask.clone()
                attn_mask[query_padding_mask] = False  # unmask those rows

            attn_mask_3d = attn_mask.repeat_interleave(self.cross.num_heads, dim=0)

        weighted_Q, att_weights = self.cross(
            Q, K, V,
            attn_mask=attn_mask_3d,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )

        x = self.norm1(Q + weighted_Q)
        x = self.norm2(x + self.ff(x))

        # prevents NaNs from leaking to pooling
        if query_padding_mask is not None:
            x = x.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)

        #detect nan
        #print(x)
        #if torch.isnan(x).any():
        #    raise RuntimeError("NaN detected in CrossAttentionNodesBlock output")
        
        return x, att_weights
    

class CrossAttentionNodesBlockv2(nn.Module):
    """
    Bidirectional cross-attention between:
      - TCR nodes (queries)
      - peptide nodes only (keys / values)

    MHC nodes are excluded by construction.
    """

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()

        # TCR -> peptide
        self.attn_tcr_to_pep = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        # peptide -> TCR
        self.attn_pep_to_tcr = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        # Transformer-style blocks
        self.norm_tcr_1 = nn.LayerNorm(dim)
        self.norm_tcr_2 = nn.LayerNorm(dim)

        self.norm_pep_1 = nn.LayerNorm(dim)
        self.norm_pep_2 = nn.LayerNorm(dim)

        self.ff_tcr = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LeakyReLU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

        self.ff_pep = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LeakyReLU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        tcr_x,                 # [B, Ltcr, D]
        pmhc_x,                # [B, Lpmhc, D]
        peptide_mask,          # [B, Lpmhc]  True = peptide
        tcr_padding_mask=None, # [B, Ltcr]   True = padding
        pmhc_padding_mask=None # [B, Lpmhc]  True = padding
    ):
        """
        Returns
        -------
        tcr_out : [B, Ltcr, D]
        pep_out : [B, Lpmhc, D]   (non-peptide positions zeroed)
        attn_tp : [B, Ltcr, Lpep]
        attn_pt : [B, Lpep, Ltcr]
        """
        print('peptide_mask:')
        print(peptide_mask)
        quit()
        # --------------------------------------------------
        # 1. Sanity check: at least one peptide per graph
        # --------------------------------------------------
        if (peptide_mask.sum(dim=1) == 0).any():
            raise RuntimeError(
                "Cross-attention undefined: a graph has zero peptide nodes."
            )

        # --------------------------------------------------
        # 2. Key padding masks
        # --------------------------------------------------
        # mask everything that is NOT peptide
        pep_key_padding_mask = ~peptide_mask  # [B, Lpmhc]

        # --------------------------------------------------
        # 3. TCR -> peptide
        # --------------------------------------------------
        tcr_attn, attn_tp = self.attn_tcr_to_pep(
            query=tcr_x,
            key=pmhc_x,
            value=pmhc_x,
            key_padding_mask=pep_key_padding_mask,
            need_weights=True
        )

        tcr_out = self.norm_tcr_1(tcr_x + tcr_attn)
        tcr_out = self.norm_tcr_2(tcr_out + self.ff_tcr(tcr_out))

        if tcr_padding_mask is not None:
            tcr_out = tcr_out.masked_fill(
                tcr_padding_mask.unsqueeze(-1), 0.0
            )

        # --------------------------------------------------
        # 4. peptide -> TCR
        # --------------------------------------------------
        pep_attn, attn_pt = self.attn_pep_to_tcr(
            query=pmhc_x,
            key=tcr_x,
            value=tcr_x,
            key_padding_mask=tcr_padding_mask,
            need_weights=True
        )

        pep_out = self.norm_pep_1(pmhc_x + pep_attn)
        pep_out = self.norm_pep_2(pep_out + self.ff_pep(pep_out))

        # explicitly zero non-peptide nodes
        pep_out = pep_out.masked_fill(
            (~peptide_mask).unsqueeze(-1), 0.0
        )

        if pmhc_padding_mask is not None:
            pep_out = pep_out.masked_fill(
                pmhc_padding_mask.unsqueeze(-1), 0.0
            )

        return tcr_out, pep_out, attn_tp, attn_pt

        
# -----------------------------
# 2. Graph encoder (GCN)
# -----------------------------
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, out_dim)
        self.conv2 = GCNConv(out_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        #x = global_mean_pool(x, batch)
        return x


class GATEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, out_dim // heads, heads=heads)
        self.conv2 = GATv2Conv(out_dim, out_dim // heads, heads=heads)
        self.dropout = dropout
    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        #x = global_mean_pool(x, batch)
        return x

class GINEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128, dropout=0.1):
        super().__init__()
        
        nn1 = nn.Sequential(
              nn.Linear(in_dim, out_dim),
              nn.BatchNorm1d(out_dim), nn.ReLU(),
              nn.Linear(out_dim, out_dim), nn.ReLU())

        nn2 = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.BatchNorm1d(out_dim), nn.ReLU(),
                nn.Linear(out_dim, out_dim), nn.ReLU())
        
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

        self.dropout = dropout

    def forward(self, x, edge_index, batch, dp=True):
        x1 = F.leaky_relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.leaky_relu(self.conv2(x1, edge_index))

        return x1, x2

class GATGINEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        
        self.gat = GATv2Conv(in_dim, out_dim // heads, heads=heads)
        self.gin = GINConv(nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU()
        ))
        self.dropout = dropout
    def forward(self, x, edge_index, batch):
        x1 = F.leaky_relu(self.gat(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.leaky_relu(self.gin(x1, edge_index))

        h = global_add_pool(x2, batch)
        return h
# -----------------------------
# 3. Main Model (Dual GCN + Cross Attention)
# -----------------------------
class CrossAttentionGCN(nn.Module):
    def __init__(self, cfg: TrainConfigs, node_features_len: int):
        super().__init__()
        
        # Model parameters
        out_dim = cfg.model_params.get('out_channels', [128])[0]  # dimension after first conv layer
        self.dropout = cfg.model_params.get('dropout', 0.2)  # dropout rate
        self.ch_names = cfg.channels  # channel names, e.g., ['TCR', 'pMHC']

        self.encoder_A = GCNEncoder(node_features_len, out_dim)
        self.encoder_B = GCNEncoder(node_features_len, out_dim)
        self.cross_attention = CrossAttentionBlock(out_dim, dropout=self.dropout)

        # self.linear_A = nn.Sequential(
        #     nn.Linear(out_dim, out_dim),
        #     nn.ReLU()
        # )
        
        # self.linear_B = nn.Sequential(
        #     nn.Linear(out_dim, out_dim),
        #     nn.ReLU()
        # )

        self.lin = nn.Sequential(
            nn.Linear(out_dim*2, out_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim, 1)
        )

    def forward(self, data):

        name1 = self.ch_names[0]
        name2 = self.ch_names[1]

        ch1_x, ch1_batch = data[name1].x.float(), data[name1].batch
        ch1_edge_index = data[(name1,"intra",name1)].edge_index.long()

        ch2_x, ch2_batch = data[name2].x.float(), data[name2].batch
        ch2_edge_index = data[(name2,"intra",name2)].edge_index.long()

        gA = self.encoder_A(ch1_x, ch1_edge_index, ch1_batch)
        gB = self.encoder_B(ch2_x, ch2_edge_index, ch2_batch)

        hA = global_mean_pool(gA, ch1_batch)
        hB = global_mean_pool(gB, ch2_batch)

        # Bidirectional cross-attention
        hA_att = self.cross_attention(hA, hB)
        hB_att = self.cross_attention(hB, hA)

        # Combine attended embeddings
        #hA_final = self.linear_A(hA + hA_att)
        #hB_final = self.linear_B(hB + hB_att)

        # Concatenate and classify
        self.concat_embed = torch.cat([hA_att, hB_att], dim=-1)

        h = F.dropout(self.concat_embed, p=self.dropout, training=self.training)
        h = self.lin(h)
        return h
    
class CrossAttentionGAT(nn.Module):
    def __init__(self, cfg: TrainConfigs, node_features_len: int):
        super().__init__()
        
        # Model parameters
        out_dim = cfg.model_params.get('out_channels', [128])[0]  # dimension after first conv layer
        self.dropout = cfg.model_params.get('dropout', 0.2)  # dropout rate
        self.ch_names = cfg.channels  # channel names, e.g., ['TCR', 'pMHC']

        self.encoder_A = GATEncoder(node_features_len, out_dim)
        self.encoder_B = GATEncoder(node_features_len, out_dim)
        self.cross_attention = CrossAttentionBlock(out_dim, dropout=self.dropout)

        # self.linear_A = nn.Sequential(
        #     nn.Linear(out_dim, out_dim),
        #     nn.ReLU()
        # )
        
        # self.linear_B = nn.Sequential(
        #     nn.Linear(out_dim, out_dim),
        #     nn.ReLU()
        # )

        self.lin = nn.Sequential(
            nn.Linear(out_dim*2, out_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim, 1)
        )

    def forward(self, data):

        name1 = self.ch_names[0]
        name2 = self.ch_names[1]

        ch1_x, ch1_batch = data[name1].x.float(), data[name1].batch
        ch1_edge_index = data[(name1,"intra",name1)].edge_index.long()

        ch2_x, ch2_batch = data[name2].x.float(), data[name2].batch
        ch2_edge_index = data[(name2,"intra",name2)].edge_index.long()
        gA = self.encoder_A(ch1_x, ch1_edge_index, ch1_batch)
        gB = self.encoder_B(ch2_x, ch2_edge_index, ch2_batch)

        hA = global_mean_pool(gA, ch1_batch)
        hB = global_mean_pool(gB, ch2_batch)

        # Bidirectional cross-attention
        hA_att = self.cross_attention(hA, hB)
        hB_att = self.cross_attention(hB, hA)

        # Combine attended embeddings
        #hA_final = self.linear_A(hA + hA_att)
        #hB_final = self.linear_B(hB + hB_att)

        # Concatenate and classify
        self.concat_embed = torch.cat([hA_att, hB_att], dim=-1)

        h = F.dropout(self.concat_embed, p=self.dropout, training=self.training)
        h = self.lin(h)
        return h


class CrossAttentionGIN(nn.Module):
    def __init__(self, cfg: TrainConfigs, node_features_len: int):
        super().__init__()
        
        # Model parameters
        out_dim = cfg.model_params.get('out_channels', [128])[0]  # dimension after first conv layer
        self.dropout = cfg.model_params.get('dropout', 0.2)  # dropout rate
        self.ch_names = cfg.channels  # channel names, e.g., ['TCR', 'pMHC']

        self.encoder_A = GINEncoder(node_features_len, out_dim)
        self.encoder_B = GINEncoder(node_features_len, out_dim)

        self.cross_attention = CrossAttentionBlock(out_dim * 2, dropout=self.dropout)

        self.lin = nn.Sequential(
            nn.Linear(out_dim*4, out_dim//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim//2, 1)
        )

    def forward(self, data):

        name1 = self.ch_names[0]
        name2 = self.ch_names[1]

        ch1_x, ch1_batch = data[name1].x.float(), data[name1].batch
        ch1_edge_index = data[(name1,"intra",name1)].edge_index.long()

        ch2_x, ch2_batch = data[name2].x.float(), data[name2].batch
        ch2_edge_index = data[(name2,"intra",name2)].edge_index.long()

        # GIN Pooling
        hA1, hA2 = self.encoder_A(ch1_x, ch1_edge_index, ch1_batch)
        hB1, hB2 = self.encoder_B(ch2_x, ch2_edge_index, ch2_batch)

        hA1 = global_add_pool(hA1, ch1_batch)
        hA2 = global_add_pool(hA2, ch1_batch)
        hB1 = global_add_pool(hB1, ch2_batch)
        hB2 = global_add_pool(hB2, ch2_batch)

        hA = torch.cat([hA1, hA2], dim=-1)
        hB = torch.cat([hB1, hB2], dim=-1)

        # Bidirectional cross-attention
        hA_att = self.cross_attention(hA, hB)
        hB_att = self.cross_attention(hB, hA)

        # Combine attended embeddings
        #hA_final = self.linear_A(hA1 + hA_att)
        #hB_final = self.linear_B(hB1 + hB_att)

        # Concatenate and classify
        self.concat_embed = torch.cat([hA_att, hB_att], dim=-1)

        h = F.dropout(self.concat_embed, p=self.dropout, training=self.training)
        h = self.lin(h)
        return h

class CrossAttentionNodesGIN(nn.Module):
    def __init__(self, cfg: TrainConfigs, node_features_len: int, max_pep_len: int = 16):
        super().__init__()
        
        # Model parameters
        out_dim = cfg.model_params.get('out_channels', [128])[0]  # dimension after first conv layer
        self.dropout = cfg.model_params.get('dropout', 0.2)  # dropout rate
        self.ch_names = cfg.channels  # channel names, e.g., ['TCR', 'pMHC']

        # positional encoding configs
        self.pe_tcr = cfg.model_params.get('pe_tcr', False)
        self.pe_epitope = cfg.model_params.get('pe_epitope', False)
        self.pe_mhc = cfg.model_params.get('pe_mhc', False)

        # positional encoders
        if self.pe_tcr:
            self.pos_enc_tcr = PositTCREncoder(embed_dim=node_features_len)

        if self.pe_epitope:
            self.pos_enc_epitope = PositEpitopeEncoder(embed_dim=node_features_len,
                                                       max_len=max_pep_len)
        if self.pe_mhc:
            self.pos_enc_mhc = PositMHCEncoder(embed_dim=node_features_len)       

        self.encoder_A = GINEncoder(node_features_len, out_dim)
        self.encoder_B = GINEncoder(node_features_len, out_dim)

        self.cross_attention = CrossAttentionNodesBlockv1(out_dim, dropout=self.dropout)

        self.lin = nn.Sequential(
            nn.Linear(out_dim*4, out_dim//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim//2, 1)
        )

    def forward(self, data):
        name1 = self.ch_names[0]
        name2 = self.ch_names[1]

        ch1_x, ch1_batch = data[name1].x.float(), data[name1].batch
        ch1_edge_index = data[(name1,"intra",name1)].edge_index.long()
        ch1_att_mask = data[name1].mask

        ch2_x, ch2_batch = data[name2].x.float(), data[name2].batch
        ch2_edge_index = data[(name2,"intra",name2)].edge_index.long()
        ch2_att_mask = data[name2].mask

        ch1_posit_tcr = data[name1].posit_tcr if self.pe_tcr else None
        ch2_posit_epitope = data[name2].posit_epitope if self.pe_epitope else None
        ch2_posit_mhc = data[name2].posit_MHC if self.pe_mhc else None

        # POSITIONAL ENCODING
        if self.pe_tcr:
            ch1_x = self.pos_enc_tcr(ch1_x, ch1_posit_tcr)
        if self.pe_epitope and self.pe_mhc:
            ch2_x = self.pos_enc_epitope(ch2_x, ch2_posit_epitope)
            ch2_x = self.pos_enc_mhc(ch2_x, ch2_posit_mhc)


        # Create attention masks
        ch1_att_mask_dense = to_dense_batch(ch1_att_mask, ch1_batch)[0]
        ch2_att_mask_dense = to_dense_batch(ch2_att_mask, ch2_batch)[0]

        # Create attention mask based on node masks -> inverted signal to deal with padding false
        att_mask = ~ch1_att_mask_dense.unsqueeze(2) | ~ch2_att_mask_dense.unsqueeze(1)   # [B, Z, W]

        # GIN 
        hA1, hA2 = self.encoder_A(ch1_x, ch1_edge_index, ch1_batch)
        hB1, hB2 = self.encoder_B(ch2_x, ch2_edge_index, ch2_batch)

        hA1_dense, maskA1 = to_dense_batch(hA1, ch1_batch)
        hA2_dense, maskA2 = to_dense_batch(hA2, ch1_batch)
        hB1_dense, maskB1 = to_dense_batch(hB1, ch2_batch)
        hB2_dense, maskB2 = to_dense_batch(hB2, ch2_batch)

        if (maskA1.sum(dim=1) == 0).any() or (maskB1.sum(dim=1) == 0).any():
            raise RuntimeError("Found an empty graph in channel B within this batch (no valid keys).")

        hA1_att, hA1_att_w  = self.cross_attention(hA1_dense, hB1_dense, hB1_dense, attention_mask=att_mask, 
                                                   query_padding_mask=~maskA1, key_padding_mask=~maskB1)
        hA2_att, hA2_att_w = self.cross_attention(hA2_dense, hB2_dense, hB2_dense, attention_mask=att_mask, 
                                                  query_padding_mask=~maskA2, key_padding_mask=~maskB2)


        hA1 = torch.sum(hA1_att, dim=1)
        hA2 = torch.sum(hA2_att, dim=1)
        
        # Channel A
        hA = torch.cat([hA1, hA2], dim=-1)
        
        # Channel B
        hB1 = global_add_pool(hB1, ch2_batch)
        hB2 = global_add_pool(hB2, ch2_batch)
        hB = torch.cat([hB1, hB2], dim=-1)

        # Concatenate and classify
        self.concat_embed = torch.cat([hA, hB], dim=-1)
        h = F.dropout(self.concat_embed, p=self.dropout, training=self.training)
        h = self.lin(h)
        return h

class DoubleCrossAttentionNodesGIN(nn.Module):
    def __init__(self, cfg: TrainConfigs, node_features_len: int):
        super().__init__()
        
        # Model parameters
        out_dim = cfg.model_params.get('out_channels', [128])[0]  # dimension after first conv layer
        self.dropout = cfg.model_params.get('dropout', 0.2)  # dropout rate
        self.ch_names = cfg.channels  # channel names, e.g., ['TCR', 'pMHC']

        self.encoder_A = GINEncoder(node_features_len, out_dim)
        self.encoder_B = GINEncoder(node_features_len, out_dim)

        self.cross_nodes_attention = CrossAttentionNodesBlockv1(out_dim, dropout=self.dropout)
        self.cross_attention = CrossAttentionBlock(out_dim * 2, dropout=self.dropout)

        self.lin = nn.Sequential(
            nn.Linear(out_dim*4, out_dim//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim//2, 1)
        )

    def forward(self, data):
        name1 = self.ch_names[0]
        name2 = self.ch_names[1]

        ch1_x, ch1_batch = data[name1].x.float(), data[name1].batch
        ch1_edge_index = data[(name1,"intra",name1)].edge_index.long()
        ch1_att_mask = data[name1].mask

        ch2_x, ch2_batch = data[name2].x.float(), data[name2].batch
        ch2_edge_index = data[(name2,"intra",name2)].edge_index.long()
        ch2_att_mask = data[name2].mask

        ch1_att_mask_dense = to_dense_batch(ch1_att_mask, ch1_batch)[0]
        ch2_att_mask_dense = to_dense_batch(ch2_att_mask, ch2_batch)[0]

        # Create attention mask based on node masks -> inverted signal to deal with padding false
        att_mask = ~ch1_att_mask_dense.unsqueeze(2) | ~ch2_att_mask_dense.unsqueeze(1)   # [B, Z, W]
        
        # GIN
        hA1, hA2 = self.encoder_A(ch1_x, ch1_edge_index, ch1_batch)
        hB1, hB2 = self.encoder_B(ch2_x, ch2_edge_index, ch2_batch)

        hA1_dense, maskA1 = to_dense_batch(hA1, ch1_batch)
        hA2_dense, maskA2 = to_dense_batch(hA2, ch1_batch)
        hB1_dense, maskB1 = to_dense_batch(hB1, ch2_batch)
        hB2_dense, maskB2 = to_dense_batch(hB2, ch2_batch)

        if (maskA1.sum(dim=1) == 0).any() or (maskB1.sum(dim=1) == 0).any():
            raise RuntimeError("Found an empty graph in channel B within this batch (no valid keys).")

        hA1_att, hA1_att_w  = self.cross_nodes_attention(hA1_dense, hB1_dense, hB1_dense, attention_mask=att_mask, 
                                                   query_padding_mask=~maskA1, key_padding_mask=~maskB1)
        hA2_att, hA2_att_w = self.cross_nodes_attention(hA2_dense, hB2_dense, hB2_dense, attention_mask=att_mask, 
                                                  query_padding_mask=~maskA2, key_padding_mask=~maskB2)


        hA1 = torch.sum(hA1_att, dim=1)
        hA2 = torch.sum(hA2_att, dim=1)
        
        # Channel A
        hA = torch.cat([hA1, hA2], dim=-1)
        
        # Channel B
        hB1 = global_add_pool(hB1, ch2_batch)
        hB2 = global_add_pool(hB2, ch2_batch)
        hB = torch.cat([hB1, hB2], dim=-1)

        # Bidirectional cross-attention
        hA_embed_att = self.cross_attention(hA, hB)
        hB_embed_att = self.cross_attention(hB, hA)

        # Concatenate and classify
        self.concat_embed = torch.cat([hA_embed_att, hB_embed_att], dim=-1)
        h = F.dropout(self.concat_embed, p=self.dropout, training=self.training)
        h = self.lin(h)
        return h
    

class XATTGraph(nn.Module):
    def __init__(self, cfg: TrainConfigs, node_features_len: int, max_pep_len: int = 16):
        super().__init__()
        
        # Model parameters
        out_dim = cfg.model_params.get('out_channels', [128])[0]  # dimension after first conv layer
        self.dropout = cfg.model_params.get('dropout', 0.2)  # dropout rate
        self.ch_names = cfg.channels  # channel names, e.g., ['TCR', 'pMHC']
        self.cross_nodes = cfg.model_params.get('cross_nodes', None) 
        self.cross_embed = cfg.model_params.get('cross_embed', None)
        graphenc = cfg.model_params.get('graph_encoder', None)

        # positional encoding configs
        self.pe_tcr = cfg.model_params.get('pe_tcr', None)
        self.pe_epitope = cfg.model_params.get('pe_epitope', None)
        self.pe_mhc = cfg.model_params.get('pe_mhc', None)
        self.pe_tcr = True
        self.pe_epitope = True
        self.pe_mhc = True
        self.cross_nodes = True
        self.cross_embed = True

        # select graph encoder
        if graphenc == 'GIN':
            graphenc = GINEncoder
        elif graphenc == 'GAT':
            graphenc = GATEncoder
        elif graphenc == 'GCN':
            graphenc = GCNEncoder
        else:
            raise ValueError(f"Unsupported graph encoder: {graphenc}")
        

        # positional encoders
        if self.pe_tcr:
            self.pos_enc_tcr = PositTCREncoder(embed_dim=node_features_len)

        if self.pe_epitope:
            self.pos_enc_epitope = PositEpitopeEncoder(embed_dim=node_features_len,
                                                       max_len=max_pep_len)
        if self.pe_mhc:
            self.pos_enc_mhc = PositMHCEncoder(embed_dim=node_features_len)

        self.encoder_A = graphenc(node_features_len, out_dim)
        self.encoder_B = graphenc(node_features_len, out_dim)

        if self.cross_nodes:
            self.cross_nodes_attention = CrossAttentionNodesBlockv2(out_dim, dropout=self.dropout)

        if self.cross_embed:
            self.cross_attention = CrossAttentionBlock(out_dim * 2, dropout=self.dropout)

        self.lin = nn.Sequential(
            nn.Linear(out_dim*4, out_dim//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim//2, 1)
        )

    def forward(self, data):

        name1 = self.ch_names[0]
        name2 = self.ch_names[1]

        ch1_x, ch1_batch = data[name1].x.float(), data[name1].batch
        ch1_edge_index = data[(name1,"intra",name1)].edge_index.long()

        ch2_x, ch2_batch = data[name2].x.float(), data[name2].batch
        ch2_resids = data[name2].resid
        ch2_edge_index = data[(name2,"intra",name2)].edge_index.long()
        print(ch2_resids)

        ch1_posit_tcr = data[name1].posit_TCR
        ch2_posit_epitope = data[name2].posit_epitope
        ch2_posit_mhc = data[name2].posit_MHC
        print(ch2_posit_epitope)
        #check if there is nan in the input features
        if torch.isnan(ch1_x).any() or torch.isnan(ch2_x).any():
            raise RuntimeError("NaN values found in input node features before positional encoding.")
              
        # POSITIONAL ENCODING
        if self.pe_tcr:
            ch1_x = self.pos_enc_tcr(ch1_x, ch1_posit_tcr)

        if self.pe_epitope:
            ch2_x = self.pos_enc_epitope(ch2_x, ch2_posit_epitope, ch2_batch)

        if self.pe_mhc:
            ch2_x = self.pos_enc_mhc(ch2_x, ch2_resids, ch2_posit_mhc)
            
        # ENCODERS

        if GINEncoder == type(self.encoder_A):
            # GIN 
            gA1, gA2 = self.encoder_A(ch1_x, ch1_edge_index, ch1_batch)
            gB1, gB2 = self.encoder_B(ch2_x, ch2_edge_index, ch2_batch)

            gA_dense, maskA = to_dense_batch(gA2, ch1_batch) # use second layer
            gB_dense, maskB = to_dense_batch(gB2, ch2_batch) # use second layer

            if (maskA.sum(dim=1) == 0).any() or (maskB.sum(dim=1) == 0).any():
                raise RuntimeError("Found an empty graph within this batch (no valid keys).")

        elif GATEncoder == type(self.encoder_A) or GCNEncoder == type(self.encoder_A):
            # GAT
            gA = self.encoder_A(ch1_x, ch1_edge_index, ch1_batch)
            gB = self.encoder_B(ch2_x, ch2_edge_index, ch2_batch)

            gA_dense, maskA = to_dense_batch(gA, ch1_batch)
            gB_dense, maskB = to_dense_batch(gB, ch2_batch)

            if (maskA.sum(dim=1) == 0).any() or (maskB.sum(dim=1) == 0).any():
                raise RuntimeError("Found an empty graph within this batch (no valid keys).")
        
        # CROSS ATTENTION ON NODES
        if self.cross_nodes:

            # maskA, maskB: [B, L] True = valid node
            # ch2_posit_epitope: [N] or [B, L] True = peptide node

            # Build peptide mask in dense form
            peptide_mask_dense = to_dense_batch(
                ch2_posit_epitope.bool(), ch2_batch
            )[0] & maskB
            # shape: [B, Lpmhc]
            print(ch2_batch)
            print(to_dense_batch(
                ch2_posit_epitope.bool(), ch2_batch
            ))
            # Safety check (CRITICAL)
            if (peptide_mask_dense.sum(dim=1) == 0).any():
                raise RuntimeError(
                    "At least one graph has zero peptide nodes — "
                    "cross-attention undefined."
                )

            # Bidirectional cross-attention
            gA_att, gB_att, gA_att_w, gB_att_w = self.cross_nodes_attention(
                tcr_x=gA_dense,              # [B, Ltcr, D]
                pmhc_x=gB_dense,             # [B, Lpmhc, D]
                peptide_mask=peptide_mask_dense,
                tcr_padding_mask=~maskA,     # True = padding
                pmhc_padding_mask=~maskB
            )
        else:
            gA_att = gA_dense
            gB_att = gB_dense

        # GRAPH POOLING
        if GINEncoder == type(self.encoder_A):
            gA2 = torch.sum(gA_att, dim=1) if self.cross_nodes else torch.sum(gA_dense, dim=1)
            gB2 = torch.sum(gB_att, dim=1) if self.cross_nodes else torch.sum(gB_dense, dim=1)

            hA = torch.cat([global_add_pool(gA1, ch1_batch), gA2], dim=-1)
            hB = torch.cat([global_add_pool(gB1, ch2_batch), gB2], dim=-1)

        elif GATEncoder == type(self.encoder_A) or GCNEncoder == type(self.encoder_A):
            hA = torch.mean(gA_att, dim=1) if self.cross_nodes else torch.mean(gA_dense, dim=1)
            hB = torch.mean(gB_att, dim=1) if self.cross_nodes else torch.mean(gB_dense, dim=1)

        # CROSS ATTENTION ON EMBEDDINGS
        if self.cross_embed:
            hA = self.cross_attention(hA, hB)
            hB = self.cross_attention(hB, hA)

        # Concatenate and classify
        self.concat_embed = torch.cat([hA, hB], dim=-1)
        h = F.dropout(self.concat_embed, p=self.dropout, training=self.training)
        h = self.lin(h)
        return h
    
class MultiGCN(nn.Module):
    """
    General-purpose GCN stack with arbitrary number of layers and customizable activation.
    """

    def __init__(self, cfg: TrainConfigs, node_features_len: int):
        super().__init__()
        
        # Model parameters
        out_dim = cfg.model_params.get('out_channels', [128])[0]  # dimension after first conv layer
        self.dropout = cfg.model_params.get('dropout', 0.2)  # dropout rate
        self.n_layers = cfg.model_params.get('n_layers', 1)  # number of GCN layers
        self.ch_names = cfg.channels  # channel names, e.g., ['TCR', 'pMHC']

        assert self.n_layers >= 1, "Number of layers must be >= 1"

        # Build GCN layers dynamically
        layers_ch1 = []
        for i in range(self.n_layers):
            input_dim = node_features_len if i == 0 else out_dim
            layers_ch1.append(GCNConv(input_dim, out_dim))
        self.layers_ch1 = nn.ModuleList(layers_ch1)

        # Build GCN layers for channel 2
        layers_ch2 = []
        for i in range(self.n_layers):
            input_dim = node_features_len if i == 0 else out_dim
            layers_ch2.append(GCNConv(input_dim, out_dim))
        self.layers_ch2 = nn.ModuleList(layers_ch2)

        self.lin = nn.Sequential(
            nn.Linear(out_dim*2, out_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim, 1)
        )

    def forward(self, data):

        name1 = self.ch_names[0]
        name2 = self.ch_names[1]

        x1, ch1_batch = data[name1].x.float(), data[name1].batch
        ch1_edge_index = data[(name1,"intra",name1)].edge_index.long()

        x2, ch2_batch = data[name2].x.float(), data[name2].batch
        ch2_edge_index = data[(name2,"intra",name2)].edge_index.long()

        for i, conv in enumerate(self.layers_ch1):
            x1 = conv(x1, ch1_edge_index)
            x1 = F.leaky_relu(x1)
            if i != self.n_layers - 1 or self.n_layers == 1:  # No activation/dropout on final layer
                x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # Graph-level readout
        x1 = global_mean_pool(x1, ch1_batch)

        for i, conv in enumerate(self.layers_ch2):
            x2 = conv(x2, ch2_edge_index)
            x2 = F.leaky_relu(x2)
            if i != self.n_layers - 1 or self.n_layers == 1:  # No activation/dropout on final layer
                x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        # Graph-level readout
        x2 = global_mean_pool(x2, ch2_batch)

        self.concat_embed = torch.cat([x1,x2], dim=-1)

        h = F.dropout(self.concat_embed, p=self.dropout, training=self.training)
        h = self.lin(h)       
        return h 

class MultiGAT(nn.Module):
    """
    General-purpose GAT stack with arbitrary number of layers and customizable activation.
    """

    def __init__(self, cfg: TrainConfigs, node_features_len: int):
        super().__init__()
        
        # Model parameters
        out_dim = cfg.model_params.get('out_channels', [128])[0]  # dimension after first conv layer
        self.dropout = cfg.model_params.get('dropout', 0.2)  # dropout rate
        self.n_layers = cfg.model_params.get('n_layers', 1)  # number of GCN layers
        self.ch_names = cfg.channels  # channel names, e.g., ['TCR', 'pMHC']
        heads = cfg.model_params.get('heads', 4)  # number of attention heads

        assert self.n_layers >= 1, "Number of layers must be >= 1"

        # Build GAT layers dynamically
        layers_ch1 = []
        for i in range(self.n_layers):
            input_dim = node_features_len if i == 0 else out_dim
            layers_ch1.append(GATv2Conv(input_dim, out_dim // heads, heads=heads))
        self.layers_ch1 = nn.ModuleList(layers_ch1)

        # Build GAT layers for channel 2
        layers_ch2 = []
        for i in range(self.n_layers):
            input_dim = node_features_len if i == 0 else out_dim
            layers_ch2.append(GATv2Conv(input_dim, out_dim // heads, heads=heads))
        self.layers_ch2 = nn.ModuleList(layers_ch2)

        self.lin = nn.Sequential(
            nn.Linear(out_dim*2, out_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim, 1)
        )

    def forward(self, data):

        name1 = self.ch_names[0]
        name2 = self.ch_names[1]

        x1, ch1_batch = data[name1].x.float(), data[name1].batch
        ch1_edge_index = data[(name1,"intra",name1)].edge_index.long()

        x2, ch2_batch = data[name2].x.float(), data[name2].batch
        ch2_edge_index = data[(name2,"intra",name2)].edge_index.long()

        for i, conv in enumerate(self.layers_ch1):
            x1 = conv(x1, ch1_edge_index)
            x1 = F.leaky_relu(x1)
            if i != self.n_layers - 1 or self.n_layers == 1:  # No activation/dropout on final layer
                x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # Graph-level readout
        x1 = global_mean_pool(x1, ch1_batch)

        for i, conv in enumerate(self.layers_ch2):
            x2 = conv(x2, ch2_edge_index)
            x2 = F.leaky_relu(x2)
            if i != self.n_layers - 1 or self.n_layers == 1:  # No activation/dropout on final layer
                x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        # Graph-level readout
        x2 = global_mean_pool(x2, ch2_batch)
        self.concat_embed = torch.cat([x1,x2], dim=-1)
        h = F.dropout(self.concat_embed, p=self.dropout, training=self.training)
        h = self.lin(h)
        return h

class MultiGIN(nn.Module):
    """
    General-purpose GIN stack with arbitrary number of layers and customizable activation.
    """

    def __init__(self, cfg: TrainConfigs, node_features_len: int):
        super().__init__()
        
        # Model parameters
        out_dim = cfg.model_params.get('out_channels', [128])[0]  # dimension after first conv layer
        self.dropout = cfg.model_params.get('dropout', 0.2)  # dropout rate
        self.n_layers = cfg.model_params.get('n_layers', 1)  # number of GCN layers
        self.ch_names = cfg.channels  # channel names, e.g., ['TCR', 'pMHC']

        assert self.n_layers >= 1, "Number of layers must be >= 1"

        # Build GIN layers dynamically
        layers_ch1 = []
        for i in range(self.n_layers):
            input_dim = node_features_len if i == 0 else out_dim
            nn_lin = nn.Sequential(nn.Linear(input_dim, out_dim),
                                    nn.BatchNorm1d(out_dim), nn.ReLU(),
                                    nn.Linear(out_dim, out_dim), nn.ReLU())
            layers_ch1.append(GINConv(nn_lin))
        self.layers_ch1 = nn.ModuleList(layers_ch1)

        # Build GIN layers for channel 2
        layers_ch2 = []
        for i in range(self.n_layers):
            input_dim = node_features_len if i == 0 else out_dim
            nn_lin = nn.Sequential(
                nn.Linear(input_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            layers_ch2.append(GINConv(nn_lin))
        self.layers_ch2 = nn.ModuleList(layers_ch2)

        self.lin = nn.Sequential(
            nn.Linear(out_dim*2*self.n_layers, out_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim, 1)
        )

    def forward(self, data):

        name1 = self.ch_names[0]
        name2 = self.ch_names[1]

        x1, ch1_batch = data[name1].x.float(), data[name1].batch
        ch1_edge_index = data[(name1,"intra",name1)].edge_index.long()

        x2, ch2_batch = data[name2].x.float(), data[name2].batch
        ch2_edge_index = data[(name2,"intra",name2)].edge_index.long()

        embed_layers_ch1 = []
        for i, conv in enumerate(self.layers_ch1):
            x1 = conv(x1, ch1_edge_index)
            x1 = F.leaky_relu(x1)
            if i != self.n_layers - 1 or self.n_layers == 1:  # No activation/dropout on final layer
                x1 = F.dropout(x1, p=self.dropout, training=self.training)
            embed_layers_ch1.append(x1)
        
        # Graph-level readout
        out_layers_ch1 = []
        for embed_layer in embed_layers_ch1:
            h1 = global_add_pool(embed_layer, ch1_batch)
            out_layers_ch1.append(h1)

        x1 = torch.cat(out_layers_ch1, dim=-1)

        embed_layers_ch2 = []
        for i, conv in enumerate(self.layers_ch2):
            x2 = conv(x2, ch2_edge_index)
            x2 = F.leaky_relu(x2)
            if i != self.n_layers - 1 or self.n_layers == 1:  # No activation/dropout on final layer
                x2 = F.dropout(x2, p=self.dropout, training=self.training)
            embed_layers_ch2.append(x2)
            
        # Graph-level readout
        out_layers_ch2 = []
        for embed_layer in embed_layers_ch2:
            h2 = global_add_pool(embed_layer, ch2_batch)
            out_layers_ch2.append(h2)

        x2 = torch.cat(out_layers_ch2, dim=-1)

        self.concat_embed = torch.cat([x1,x2], dim=-1)
        h = F.dropout(self.concat_embed, p=self.dropout, training=self.training)
        h = self.lin(h)       
        return h