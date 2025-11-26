#!/usr/bin/env python3

# local toolkit
from unittest import loader
from ..utils.registry import register, get
from ..utils.base import TrainConfigs, SamplePair, ChannelsInput
from ..utils.utils import multiproc
from .datasets import TCRpMHCDataset

# torch toolkit
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset as Dataset_n
#from torch_geometric.data import Data
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.nn import CrossEntropyLoss
#from torch_geometric.nn import global_add_pool

# third-parties
import logging as log



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()

    def forward(self):
        pass

class EmbeddingGenerator(nn.Module):
    def __init__(self, dataset: TCRpMHCDataset):
        self.dataset = dataset

    def forward(self, x):
        # Implement embedding generation logic here
        pass

class ProteinFeatures(nn.Module):
    def __init__(self, dataset: TCRpMHCDataset, config: TrainConfigs):
        
        super().__init__()
        
        log.info('Building Protein Features inputs')
        
        self.cfg = TrainConfigs(**config)
        self._convertor = get(f"loader.{self.cfg.graph_method}")(config = self.cfg)
        self._dataset   = dataset
        self._dataset   = multiproc(self._convertor, self._dataset)

    def forward(self, sample_id: str):
        pass