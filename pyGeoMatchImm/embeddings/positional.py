import math
from attr import attrs
from ..utils.base import PairsAnnotation, _AHO_CDR_RANGES, _MHC_RES_SELECTION
from ..utils.utils import multiproc
import networkx as nx
import torch
import torch.nn as nn
import numpy as np

import os
import logging

log = logging.getLogger(__name__)


class PositionalEmbedder:
    """
    Wrapper class to handle positional encoding for different chain types.
    """

    def __init__(self):
        self.tcr_embedder = PositionalTCREmbedder(allowed_chains=[PairsAnnotation.get_chain('TRA'),
                                                                PairsAnnotation.get_chain('TRB')])
        self.epi_embedder = PositionalEpiEmbedder(chain=PairsAnnotation.get_chain('epitope'))
        self.MHC_embedder = PositionalMHCEmbedder(chain=PairsAnnotation.get_chain('MHCseq'))

    def __call__(self, graph: nx.Graph, chain_name: str) -> nx.Graph:
        """
        graph: input graph with 'residue_number' node attribute
        chain_name: one of 'TCR', 'epitope', 'MHCseq'
        """

        metadata = [(data['residue_number'], data['chain_id'], gid) for gid, data in graph.nodes(data=True)]

        match chain_name:
            case 'TCR':
                pos_attr = self.tcr_embedder(metadata)

            case 'MHC':
                pos_attr = self.MHC_embedder(metadata)

            case 'epitope':
                pos_attr = self.epi_embedder(metadata)
            case _:
                raise ValueError(f"Unknown chain type for positional encoding: {chain_name}")

        nx.set_node_attributes(graph, pos_attr, name=f"posit_{chain_name}")

        return graph
    
class PositionalTCREmbedder:
    """
    Encodes AHO residue indices + chain identity into a discrete positional code.
    Output range: [0, num_embeddings - 1]
    """

    def __init__(self, allowed_chains: list):
        self.allowed_chains = allowed_chains
        self._build_cdr_lookup()

    def _build_cdr_lookup(self):
        # Default: 4 = framework
        max_resid = max(max(v) for v in _AHO_CDR_RANGES.values())
        lut = torch.full((max_resid + 1,), 4, dtype=torch.long)

        for i, (_, residues) in enumerate(_AHO_CDR_RANGES.items()):
            lut[residues] = i

        self.cdr_lut = lut


        # Chain offsets
        self.chain_offsets = {
            'D': 0,
            'E': 5
        }

    def __call__(self, resid_chains: tuple) -> torch.Tensor:
        """
        resids: (N,) tensor of AHO residue indices
        chains: list[str] of length N
        """
        resids, chains, gids = zip(*resid_chains)

        #exclude non-allowed chains
        mask = np.isin(np.array(chains), self.allowed_chains)
        resids = np.array(resids)[mask]
        chains = np.array(chains)[mask]
        gids = np.array(gids)[mask]

        #check offsets
        assert all(chain in self.chain_offsets.keys() for chain in self.allowed_chains), \
            f"Allowed chains must be in {list(self.chain_offsets.keys())}"
        
        lut = self.cdr_lut

        # CDR class lookup
        cdr_ids = lut[resids]

        # Chain offsets
        try:
            chain_offsets = torch.tensor(
                [self.chain_offsets[c] for c in chains],
                dtype=torch.long
            )
        except KeyError as e:
            raise ValueError(f"Unknown chain ID: {e}")

        encoded = cdr_ids + chain_offsets

        attr = {gid: code.item() for gid, code in zip(gids, encoded)}
        return attr

class PositTCREncoder(torch.nn.Module):
    def __init__(self, num_embeddings=10, embed_dim=5, dropout=0.1):
        super(PositTCREncoder, self).__init__()
        self.embedder = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, resids_positional_encoded: torch.Tensor) -> torch.Tensor:
        """
        x: [N, embed_dim]
        resids_positional_encoded: [N]
        """

        if resids_positional_encoded.min() < 0 or \
        resids_positional_encoded.max() >= self.embedder.num_embeddings:
            raise RuntimeError(
                f"TCR positional index out of bounds: "
                f"[{resids_positional_encoded.min()}, "
                f"{resids_positional_encoded.max()}], "
                f"num_embeddings={self.embedder.num_embeddings}"
            )
        
        if torch.isnan(self.embedder.weight).any():
            print("Embedding weights corrupted before forward pass")
        assert x.size(-1) == self.embedder.embedding_dim, \
            "Embedding dim must match x feature dim"
        
        
        x = x.clone()
        pe_embed = self.embedder(resids_positional_encoded)
        x =  x + self.dropout(pe_embed)

        return x

class PositionalEpiEmbedder:
    """
    Encodes epitope residue indices into a boolean code.
    """

    def __init__(self, chain = 'C'):
        self.chain = chain

    def __call__(self, resid_chains: tuple) -> torch.Tensor:
        """
        resids: (N,) tensor of pMHC residue indices (0-based)
        """
        _, chains, gids = zip(*resid_chains)
        epi_mask = (np.array(chains) == self.chain)
        attr = {gid: int(mask) for gid, mask in zip(gids, epi_mask)}        
        return attr

class PositEpitopeEncoder(torch.nn.Module): #implementation in pytorch = https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 16):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        half_dim = embed_dim // 2 # even embedding dimension - shortcut for simplicity
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(half_dim) * (-math.log(1000.0) / half_dim)) # 1000.0 positional resolution for short peptides
        pe = torch.zeros(max_len, embed_dim)

        pe[:, 0:2*half_dim:2] = torch.sin(position * div_term)
        pe[:, 1:2*half_dim:2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        x:     [N, D]
        mask:  [N]      (True for epitope residues)
        batch: [N]      (graph id per node)
        """

        x = x.clone()

        # Iterate over graphs (order-safe)
        for g in batch.unique(sorted=True):
            g_nodes = (batch == g)
            g_mask = g_nodes & mask

            pep_len = g_mask.sum().item()

            assert pep_len <= self.pe.size(0), \
                f"Epitope length {pep_len} exceeds max_len {self.pe.size(0)}"

            # IMPORTANT:
            # g_mask keeps the ORIGINAL node order
            pe = self.dropout(self.pe[:pep_len])
            x[g_mask.bool()] += pe

        return x
        
    
class PositionalMHCEmbedder:
    """
    Encodes MHC residue indices into a discrete positional code.
    """

    def __init__(self, chain = 'A'):
        self.chain = chain

    def __call__(self, resid_chains: tuple) -> torch.Tensor:
        """
        resids: (N,) tensor of pMHC residue indices (0-based)
        """
        resids, chains, gids = zip(*resid_chains)
        mhc_mask = (np.array(chains) == self.chain)

        attr = {gid: int(mask) for gid, mask in zip(gids, mhc_mask)}
        return attr

class PositMHCEncoder(nn.Module):
    def __init__(self, num_embeddings=2, embed_dim=5, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedder = nn.Embedding(num_embeddings, embed_dim)
        #self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, resids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:      [N, embed_dim]
        resids: [N] (residue indices)
        mask:   [N] (True for MHC residues)
        """

        assert x.size(-1) == self.embedder.embedding_dim, \
            "Embedding dim must match x feature dim"
        
        mask = mask.bool()

        # avoid in-place corruption
        mhc_resids = resids[mask].clone().long()

        # Map to helix IDs
        mhc_ids = torch.zeros_like(mhc_resids)
        mhc_ids[mhc_resids >= 94] = 1

        # Embed
        mhc_embed = self.embedder(mhc_ids)

        # Add to x
        x = x.clone()
        mhc_embed = self.dropout(mhc_embed)
        x[mask] = x[mask] + mhc_embed
        
        return x