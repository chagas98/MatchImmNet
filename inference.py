#!/usr/bin/env python3

from pyGeoMatchImm.data.datasets import TCRpMHCDataset, ChannelsPairDataset, ChannelsGraph
from pyGeoMatchImm.test.test import Tester
from pyGeoMatchImm.utils.utils import validate_data
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from torch.nn.functional import binary_cross_entropy_with_logits
from torch_geometric.data import Data, Batch
from pathlib import Path
import os
from copy import deepcopy
import logging as log


# logger
log.basicConfig(level=log.INFO)
log.getLogger("pyGeoMatchImm").setLevel(log.INFO)
log.getLogger("MDAnalysis").setLevel(log.WARNING)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_params = {
}

train_params = {
    "batch_size"      : 16,
}

config = {
    "source"          : "pdb",
    "channels"        : ["TCR", "pMHC"],
    "pairing_method"  : "basic",
    "embed_method"    : ["esm3"],
    "graph_method"    : "graphein",
    "negative_prop"   : 3,
    "edge_params"     : ["distance_threshold"],
    "node_params"     : ["amino_acid_one_hot", "hbond_donors", "hbond_acceptors", "dssp_config"],
    "graph_params"    : ["rsa"],
    "other_params"    : {"granularity": "centroids"},
    "dist_threshold"  : 8.0,
    "concat_embed"    : "all",
    "train_params"    : train_params,
    "model_params"    : model_params,
    "save_dir"        : ''
}

map_cols = {
    'TCR_ID': 'id',
    'TRA_ref': 'TRA',
    'TRB_ref': 'TRB',
    'MHCseq_ref': 'MHCseq',
    'assigned_allele': 'mhc_allele',
    'peptide': 'epitope'
}

# Load Test Data
test_path = "data/02-processed/tcr3d_20251004_renamed.csv"
test_data = pd.read_csv(test_path, index_col=0)
test_data.rename(columns=map_cols, inplace=True)
test_data.reset_index(drop=True, inplace=True)

test_data = test_data.dropna(subset=['filepath_a', 'filepath_b'], how='any')

select_columns = ['id', 'TRA', 'TRB', 'CDR1A', 'CDR2A', 'CDR3A', 'CDR1B', 'CDR2B', 'CDR3B', 'TRA_num', 'TRB_num', 'epitope', 'MHCseq', 'mhc_allele', 'filepath_a', 'filepath_b', 'label', 'source']
test_data.drop_duplicates(subset=["TRA", "TRB", "epitope", "MHCseq"], inplace=True)
test_data = test_data[select_columns].copy()

test_data.to_csv("data/02-processed/tcrpMHC_combined_test_data.csv", index=False)

print(f"Positives test data size: {test_data.shape}")

dataset = TCRpMHCDataset("data/02-processed/tcrpMHC_combined_train_data.csv", config=config)
embed = config['embed_method'][0]

log.info("Using embedding method: %s", embed)
log.info("Generating graphs...")

channels = ChannelsGraph(dataset, config=config, embed_method=embed)
peptides = [ch.get('peptide') for ch in channels]

ds = ChannelsPairDataset(
    channels_graph=channels,
    ch1_name=channels.channel_names[0],
    ch2_name=channels.channel_names[1],
)

chids, chlabels, chpeptides, chtras, chtrbs = zip(*[(i.get('id'), i.get('label'), i.get('peptide'), i.get('TRA'), i.get('TRB')) for i in channels])

validate_data(test_data, dict_lists={
    "id": chids,
    "epitope": chpeptides,
    "label": chlabels,
    "TRA": chtras,
    "TRB": chtrbs
}, cols_to_check=["epitope", "label", "TRA", "TRB"])

model_path = "/home/samuel.assis/MatchImm/MatchImmNet/developments/increase_dataset_1217_neg3_bs16_lr1.0_v3/cangin_neg3_atchley_dp03/model_ASNENMETM.pt"
model_class = torch.load(model_path, weights_only=False)
Tester(model_class, device, configs=config).predict(ds)