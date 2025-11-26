# !/usr/bin/env python3

# local
from ..utils.base import TrainConfigs

# third-parties
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, GATConv, GATv2Conv, GraphNorm,
                                global_max_pool as gmp, 
                                global_add_pool as gap,
                                global_mean_pool as gep,
                                global_sort_pool as gsp)

import logging
log = logging.getLogger(__name__)

torch.manual_seed(42); np.random.seed(42)

class Jha_GCN(nn.Module):
    def __init__(self, cfg: TrainConfigs, node_features_len: int):
        super().__init__()
        
        # Model parameters
        output_dim = cfg.model_params.get('out_channels', [64])[0]  # dimension after first conv layer
        n_output = cfg.model_params.get('n_output', 1)  # final output dimension
        dropout = cfg.model_params.get('dropout', 0.2)  # dropout rate
        self.ch_names = cfg.channels  # channel names, e.g., ['TCR', 'pMHC']

        # for channel 1
        self.ch1_conv1 = GCNConv(node_features_len, node_features_len * 5)
        self.ch1_fc1 = nn.Linear(node_features_len * 5, output_dim)

        # for channel 2
        self.ch2_conv1 = GCNConv(node_features_len, node_features_len * 5)
        self.ch2_fc1 = nn.Linear(node_features_len * 5, output_dim)

        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, 16)
        self.out = nn.Linear(16, n_output)

    def forward(self, data):

        name1 = self.ch_names[0]
        name2 = self.ch_names[1]

        ch1_x, ch1_batch = data[name1].x.float(), data[name1].batch
        ch1_edge_index = data[(name1,"intra",name1)].edge_index.long()

        ch2_x, ch2_batch = data[name2].x.float(), data[name2].batch
        ch2_edge_index = data[(name2,"intra",name2)].edge_index.long()

        # ------------ CHANNEL 1 -------------
        x = self.ch1_conv1(ch1_x, ch1_edge_index)

        #print(x)
        x = F.leaky_relu(x)

	    # global pooling
        x = gep(x, ch1_batch)

        # flatten
        x = F.leaky_relu(self.ch1_fc1(x))
        x = self.dropout(x)
    
        # ------------ CHANNEL 2 -------------
        xt = self.ch2_conv1(ch2_x, ch2_edge_index)
        xt = F.leaky_relu(xt)

	    # global pooling
        xt = gep(xt, ch2_batch)

        # flatten
        xt = F.leaky_relu(self.ch2_fc1(xt))
        xt = self.dropout(xt)

    	# Concatenation  
        self.concat_embed = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(self.concat_embed)
        xc = F.leaky_relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = F.leaky_relu(xc)
        xc = self.dropout(xc)
        self.posmlp = xc
        out = self.out(xc)

        return out



class Jha_GAT(nn.Module):
    def __init__(self, cfg: TrainConfigs, node_features_len: int):
        super().__init__()

        # Model parameters
        output_dim = cfg.model_params.get('out_channels', [128])[0]  # dimension after first conv layer
        n_output = cfg.model_params.get('n_output', 1)  # final output dimension
        dropout = cfg.model_params.get('dropout', 0.2)  # dropout rate
        heads = cfg.model_params.get('heads', 4)  # number of attention heads
        hidden = cfg.model_params.get('hidden', 32)  # hidden dimension
        self.ch_names = cfg.channels  # channel names, e.g., ['TCR', 'pMHC']
        
        
        # for protein 1
        self.pro1_conv1 = GATv2Conv(node_features_len, output_dim // heads, heads=heads, dropout=dropout)
        self.pro1_fc1 = nn.Linear(output_dim, output_dim)

        # for protein 2
        self.pro2_conv1 = GATv2Conv(node_features_len, output_dim // heads, heads=heads, dropout=dropout)
        self.pro2_fc1 = nn.Linear(output_dim, output_dim)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 2 * output_dim // 2)
        self.fc2 = nn.Linear(2 * output_dim // 2, 16)
        self.out = nn.Linear(16, n_output)

    def forward(self, data):

        name1 = self.ch_names[0]
        name2 = self.ch_names[1]

        ch1_x, ch1_batch = data[name1].x.float(), data[name1].batch
        ch1_edge_index = data[(name1,"intra",name1)].edge_index.long()

        ch2_x, ch2_batch = data[name2].x.float(), data[name2].batch
        ch2_edge_index = data[(name2,"intra",name2)].edge_index.long()

    
        # Protein 1
        x = self.pro1_conv1(ch1_x, ch1_edge_index)
        x = self.relu(x)
        x = gep(x, ch1_batch)
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)

        # Protein 2
        xt = self.pro2_conv1(ch2_x, ch2_edge_index)
        xt = self.relu(xt)
        xt = gep(xt, ch2_batch)
        xt = self.relu(self.pro2_fc1(xt))
        xt = self.dropout(xt)

        # Concatenate and dense layers
        self.concat_embed = torch.cat((x, xt), dim=1)
        xc = self.relu(self.fc1(self.concat_embed))
        xc = self.dropout(xc)
        xc = self.relu(self.fc2(xc))
        self.posmlp = xc
        xc = self.dropout(xc)
        out = self.out(xc)

        return out
