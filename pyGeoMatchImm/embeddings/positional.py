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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEmbedder:
    """
    Wrapper class to handle positional encoding for different chain types.
    """

    def __init__(self, max_epitope_length: int = 14):
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

                allowed_chains = self.tcr_embedder.allowed_chains
                pos_encoded = self.tcr_embedder(metadata)

            case 'MHC':
                allowed_chains = [PairsAnnotation.get_chain('MHCseq')]
                pos_encoded = self.MHC_embedder(metadata)
            
            case 'epitope':
                allowed_chains = [PairsAnnotation.get_chain('epitope')]
                pos_encoded = self.epi_embedder(metadata)
            case _: 
                raise ValueError(f"Unknown chain type for positional encoding: {chain_name}")
        print(pos_encoded)
        mask = np.isin(np.array(chains), allowed_chains)
        attrs = {n: pos for n, pos, m in zip(ids, pos_encoded, mask) if m}
        print(attrs)
        nx.set_node_attributes(graph, attrs, name=f"posit_{chain_name}")

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
        resids, chains = zip(*resid_chains)

        #exclude non-allowed chains
        mask = np.isin(np.array(chains), self.allowed_chains)
        resids = np.array(resids)[mask]
        chains = np.array(chains)[mask]
        
        #check offsets
        assert all(chain in self.chain_offsets.keys() for chain in self.allowed_chains), \
            f"Allowed chains must be in {list(self.chain_offsets.keys())}"
        
        lut = self.cdr_lut

        # CDR class lookup
        cdr_ids = lut[resids]
        print(cdr_ids)
        print(f'resids{resids}')
        # Chain offsets
        try:
            chain_offsets = torch.tensor(
                [self.chain_offsets[c] for c in chains],
                dtype=torch.long
            )
        except KeyError as e:
            raise ValueError(f"Unknown chain ID: {e}")

        encoded = cdr_ids + chain_offsets
        return encoded

class PositTCREncoder(torch.nn.Module):
    def __init__(self, num_embeddings=10, embed_dim=5):
        super(PositTCREncoder, self).__init__()
        self.embedder = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)

    def forward(self, resids_positional_encoded: torch.Tensor) -> torch.Tensor:
        x =  self.embedder(resids_positional_encoded)
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
        _, chains = zip(*resid_chains)
        epi_mask = (np.array(chains) == self.chain)        
        return epi_mask
    
class PositEpitopeEncoder(torch.nn.Module):
    def __init__(self, d_model):
        super(PositEpitopeEncoder, self).__init__()
        i = torch.arange(0, d_model, 2, dtype=torch.float32)

        div_term = torch.exp(-torch.log(torch.tensor(1000.0)) * i / d_model) # Changed 10000 to 1000 for variation
        self.register_buffer("div_term", div_term)

    def forward(self, positions_len):
        """Compute sinusoidal positional encodings.
        Args:
            positions_len (int): Length of the positions sequence.
        Returns:
            torch.Tensor: Positional encodings of shape (seq_len, d_model)."""
        
        positions = torch.arange(0, positions_len).unsqueeze(1)
        pe = torch.zeros(positions.size(0), self.div_term.size(0) * 2,
                         device=positions.device)
        
        pe[:, 0::2] = torch.sin(positions * self.div_term)
        pe[:, 1::2] = torch.cos(positions * self.div_term)
        
        return pe
    
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
        resids, chains = zip(*resid_chains)
        mhc_mask = (np.array(chains) == self.chain)
        resid_mhc = np.array(resids)[mhc_mask]
        # 0 is alphahelice 1 and 1 is alphahelice 2
        resid_mhc[resid_mhc < 94] = 0
        resid_mhc[resid_mhc >= 94] = 1
        
        return resid_mhc

class PositMHCEncoder(torch.nn.Module):
    def __init__(self, num_embeddings=2, embed_dim=5):
        super(PositMHCEncoder, self).__init__()
        self.embedder = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)

    def forward(self, resids_positional_encoded: torch.Tensor) -> torch.Tensor:
        return self.embedder(resids_positional_encoded)
    

