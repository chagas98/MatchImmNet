import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, global_mean_pool, global_add_pool, GraphNorm
from pyGeoMatchImm.utils.base import TrainConfigs

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
        x = global_mean_pool(x, batch)
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
        x = global_mean_pool(x, batch)
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

        x1 = global_add_pool(x1, batch)
        x2 = global_add_pool(x2, batch)

        h  = torch.cat([x1, x2], dim=-1)
        return h

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

        hA = self.encoder_A(ch1_x, ch1_edge_index, ch1_batch)
        hB = self.encoder_B(ch2_x, ch2_edge_index, ch2_batch)

        # Bidirectional cross-attention
        hA_att = self.cross_attention(hA, hB)
        hB_att = self.cross_attention(hB, hA)

        # Combine attended embeddings
        #hA_final = self.linear_A(hA + hA_att)
        #hB_final = self.linear_B(hB + hB_att)

        # Concatenate and classify
        self.concat_embed = torch.cat([hA + hA_att, hB + hB_att], dim=-1)

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
        hA = self.encoder_A(ch1_x, ch1_edge_index, ch1_batch)
        hB = self.encoder_B(ch2_x, ch2_edge_index, ch2_batch)

        # Bidirectional cross-attention
        hA_att = self.cross_attention(hA, hB)
        hB_att = self.cross_attention(hB, hA)

        # Combine attended embeddings
        #hA_final = self.linear_A(hA + hA_att)
        #hB_final = self.linear_B(hB + hB_att)

        # Concatenate and classify
        self.concat_embed = torch.cat([hA + hA_att, hB + hB_att], dim=-1)

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
        
        #self.linear_A = nn.Sequential(
        #    nn.Linear(out_dim*2, out_dim),
        #    nn.BatchNorm1d(out_dim),
        #    nn.ReLU(),
        #    nn.Dropout(self.dropout)
        #)
        
        #self.linear_B = nn.Sequential(
        #    nn.Linear(out_dim*2, out_dim),
        #    nn.BatchNorm1d(out_dim),
        #    nn.ReLU(),
        #    nn.Dropout(self.dropout)
        #)

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

        hA1 = self.encoder_A(ch1_x, ch1_edge_index, ch1_batch)
        hB1 = self.encoder_B(ch2_x, ch2_edge_index, ch2_batch)

        # Bidirectional cross-attention
        hA_att = self.cross_attention(hA1, hB1)
        hB_att = self.cross_attention(hB1, hA1)

        # Combine attended embeddings
        #hA_final = self.linear_A(hA1 + hA_att)
        #hB_final = self.linear_B(hB1 + hB_att)

        # Concatenate and classify
        self.concat_embed = torch.cat([hA1 + hA_att, hB1 + hB_att], dim=-1)

        h = F.dropout(self.concat_embed, p=self.dropout, training=self.training)
        h = self.lin(h)
        return h

class CrossAttentionGATGIN(nn.Module):
    def __init__(self, cfg: TrainConfigs, node_features_len: int):
        super().__init__()
        
        # Model parameters
        out_dim = cfg.model_params.get('out_channels', [128])[0]  # dimension after first conv layer
        self.dropout = cfg.model_params.get('dropout', 0.2)  # dropout rate
        self.ch_names = cfg.channels  # channel names, e.g., ['TCR', 'pMHC']

        self.encoder_A = GATGINEncoder(node_features_len, out_dim)
        self.encoder_B = GATGINEncoder(node_features_len, out_dim)

        self.cross_attention = CrossAttentionBlock(out_dim, dropout=self.dropout)
        
        self.linear_A = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )
        
        self.linear_B = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

        self.lin = nn.Linear(out_dim*2, 1)

    def forward(self, data):
        name1 = self.ch_names[0]
        name2 = self.ch_names[1]

        ch1_x, ch1_batch = data[name1].x.float(), data[name1].batch
        ch1_edge_index = data[(name1,"intra",name1)].edge_index.long()

        ch2_x, ch2_batch = data[name2].x.float(), data[name2].batch
        ch2_edge_index = data[(name2,"intra",name2)].edge_index.long()

        hA1 = self.encoder_A(ch1_x, ch1_edge_index, ch1_batch)
        hB1 = self.encoder_B(ch2_x, ch2_edge_index, ch2_batch)

        # Bidirectional cross-attention
        hA_att = self.cross_attention(hA1, hB1)
        hB_att = self.cross_attention(hB1, hA1)

        # Combine attended embeddings
        hA_final = self.linear_A(hA1 + hA_att)
        hB_final = self.linear_B(hB1 + hB_att)

        # Concatenate and classify
        self.concat_embed = torch.cat([hA_final, hB_final], dim=-1)

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