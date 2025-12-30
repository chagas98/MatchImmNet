from ..utils.base import _AHO_CDR_RANGES
import networkx as nx
import torch

import os
import logging

log = logging.getLogger(__name__)


class PositionalCDREmbedder:
    def __init__(self):
        
        self.define_cdr()

    def define_cdr(self):
        # Create a lookup table for AHO CDR residues (mapping from 0 to 3 (CDR1, CDR2, CDR3, CDR2.5))
        self.aho_cdr_mapping = {}
 
        for i, (cdr_name, residues) in enumerate(_AHO_CDR_RANGES.items()):

            self.aho_cdr_mapping.update(dict(zip(residues, [i] * len(residues))))

    def __call__(self, resids: torch.Tensor, chains: list) -> torch.Tensor:
        
        # use chains and resids to get positional embeddings
        chains_map = {'D': 0, 'E': 5}
        chains_indices = [chains_map.get(val, -1) for val in chains]
        chains_indices = torch.tensor(chains_indices)

        resids_indices = resids.cpu().apply_(lambda val: self.aho_cdr_mapping.get(val, 4))

        resids_indices = resids_indices + chains_indices
        resids_indices = resids_indices.unsqueeze(1).to(resids.device)
        return resids_indices
    

class PositionalCDREncoder(torch.nn.Module):
    def __init__(self, num_embeddings=10, embed_dim=10):
        super(PositionalCDREncoder, self).__init__()
        self.embedder = PositionalCDREmbedder(num_embeddings=num_embeddings, embed_dim=embed_dim)

    def forward(self, resids: torch.Tensor) -> torch.Tensor:
        return self.embedder(resids)

