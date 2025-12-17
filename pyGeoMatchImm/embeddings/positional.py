from ..utils.base import _AHO_CDR_RANGES
import networkx as nx
import torch

import os
import logging

log = logging.getLogger(__name__)


class PositionalCDREmbedder:
    def __init__(self, num_embeddings=9, embed_dim=9):
        self.position_embeddings = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)
        self.define_cdr()

    def define_cdr(self):
        # Create a lookup table for AHO CDR residues (mapping from 0 to 3 (CDR1, CDR2, CDR3, CDR2.5))
        self.aho_cdr_mapping = {}
 
        for i, (cdr_name, residues) in enumerate(_AHO_CDR_RANGES.items()):

            self.aho_cdr_mapping.update(dict(zip(residues, [i] * len(residues))))

    def __call__(self, resids: torch.Tensor, chains: list) -> torch.Tensor:
        
        # use chains and resids to get positional embeddings
        chains_map = {'D': 0, 'E': 4}
        chains_indices = [chains_map.get(val, -1) for val in chains]
        chains_indices = torch.tensor(chains_indices)
        print(chains_indices)
        resids_indices = resids.cpu().apply_(lambda val: self.aho_cdr_mapping.get(val, -1))

        print('resids_indices:', resids_indices)
        resids_indices = resids_indices + chains_indices

        print('resids_indices after adding chains:', resids_indices)
        resids_indices = resids_indices.unsqueeze(1).to(resids.device)
        embeddings = self.position_embeddings(resids_indices)
        return embeddings