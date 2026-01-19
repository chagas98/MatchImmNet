from ..utils.base import _AHO_CDR_RANGES
import networkx as nx
import torch
import torch.nn as nn

import os
import logging

log = logging.getLogger(__name__)


class PositionalCDREmbedder:
    """
    Encodes AHO residue indices + chain identity into a discrete positional code.
    Output range: [0, num_embeddings - 1]
    """

    def __init__(self, device=None):
        self.device = device
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

    def __call__(self, resids: torch.Tensor, chains: list[str]) -> torch.Tensor:
        """
        resids: (N,) tensor of AHO residue indices
        chains: list[str] of length N
        """

        device = resids.device
        lut = self.cdr_lut.to(device)

        # CDR class lookup
        cdr_ids = lut[resids]

        # Chain offsets
        try:
            chain_offsets = torch.tensor(
                [self.chain_offsets[c] for c in chains],
                device=device,
                dtype=torch.long
            )
        except KeyError as e:
            raise ValueError(f"Unknown chain ID: {e}")

        encoded = cdr_ids + chain_offsets

        return encoded

    

class PositionalCDREncoder(torch.nn.Module):
    def __init__(self, num_embeddings=10, embed_dim=10):
        super(PositionalCDREncoder, self).__init__()
        self.embedder = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)

    def forward(self, resids_positional_encoded: torch.Tensor) -> torch.Tensor:
        return self.embedder(resids_positional_encoded)

