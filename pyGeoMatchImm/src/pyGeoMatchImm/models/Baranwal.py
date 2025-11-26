# !/usr/bin/env python3

# local
from ..utils.base import TrainConfigs

# third-parties
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import (GCNConv, GATConv,
                                global_max_pool as gmp, 
                                global_add_pool as gap,
                                global_mean_pool as gep,
                                global_sort_pool as gsp)

import logging
log = logging.getLogger(__name__)


class ProteinProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(ProteinProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.W_gnn             = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W1_attention      = nn.Linear(dim, dim)
        self.W2_attention      = nn.Linear(dim, dim)
        self.w                 = nn.Parameter(torch.zeros(dim))
        
        self.W_out             = nn.Linear(2*dim, 2)
        
    def gnn(self, xs1, A1, xs2, A2):
        for i in range(layer_gnn):
            hs1 = torch.relu(self.W_gnn[i](xs1))            
            hs2 = torch.relu(self.W_gnn[i](xs2))
            
            xs1 = torch.matmul(A1, hs1)
            xs2 = torch.matmul(A2, hs2)
        
        return xs1, xs2
        
    
    def mutual_attention(self, h1, h2):
        x1 = self.W1_attention(h1)
        x2 = self.W2_attention(h2)
        
        m1 = x1.size()[0]
        m2 = x2.size()[0]
        
        c1 = x1.repeat(1,m2).view(m1, m2, dim)
        c2 = x2.repeat(m1,1).view(m1, m2, dim)

        d = torch.tanh(c1 + c2)
        alpha = torch.matmul(d,self.w).view(m1,m2)
        
        b1 = torch.mean(alpha,1)
        p1 = torch.softmax(b1,0)
        s1 = torch.matmul(torch.t(x1),p1).view(-1,1)
        
        b2 = torch.mean(alpha,0)
        p2 = torch.softmax(b2,0)
        s2 = torch.matmul(torch.t(x2),p2).view(-1,1)
        
        return torch.cat((s1,s2),0).view(1,-1), p1, p2
    
    def forward(self, inputs):

        fingerprints1, adjacency1, fingerprints2, adjacency2 = inputs
        
        """Protein vector with GNN."""
        x_fingerprints1        = self.embed_fingerprint(fingerprints1)
        x_fingerprints2        = self.embed_fingerprint(fingerprints2)
        
        x_protein1, x_protein2 = self.gnn(x_fingerprints1, adjacency1, x_fingerprints2, adjacency2)
        
        """Protein vector with mutual-attention."""
        y, p1, p2     = self.mutual_attention(x_protein1, x_protein2)
        z_interaction = self.W_out(y)

        return z_interaction, p1, p2
    
    def __call__(self, data, train=True):
        
        inputs, t_interaction = data[:-1], data[-1]
        z_interaction, p1, p2 = self.forward(inputs)
        
        if train:
            loss = F.cross_entropy(z_interaction, t_interaction)
            return loss
        else:
            z = F.softmax(z_interaction, 1).to('cpu').data[0].numpy()
            t = int(t_interaction.to('cpu').data[0].numpy())
            return z, t, p1, p2
