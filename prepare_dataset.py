#!/usr/bin/env python3

from pyGeoMatchImm.data.datasets import TCRpMHCDataset, ChannelsPairDataset, ChannelsGraph
from pyGeoMatchImm.train.train import CrossValidator
from pyGeoMatchImm.models.Jha import Jha_GCN, Jha_GAT
from pyGeoMatchImm.models.models import (XATTGraph,
                                         CrossAttentionGCN,
                                         CrossAttentionGAT,
                                         CrossAttentionGIN,
                                         CrossAttentionNodesGIN)
 
from pyGeoMatchImm.utils.utils import validate_data
from pyGeoMatchImm.metrics.train_metrics import (plot_precision_recall_curve,
                                                 plot_roc_curve,
                                                 plot_loss_curve_logocv)

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from torch.nn.functional import binary_cross_entropy_with_logits
from torch_geometric.data import Data, Batch
from pathlib import Path
import os
import json
import logging as log
from itertools import product
from copy import deepcopy
import gc

# logger
log.basicConfig(level=log.INFO)
log.getLogger("pyGeoMatchImm").setLevel(log.INFO)
log.getLogger("MDAnalysis").setLevel(log.WARNING)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_structural_data(tcr3d_path: str = None,
                         af_score3_path: str = None,
                         af_score2_path: str = None,
                         id10x:list = []):
    map_cols = {
        'TCR_ID': 'id',
        'TRA_ref': 'TRA',
        'TRB_ref': 'TRB',
        'MHCseq_ref': 'MHCseq',
        'assigned_allele': 'mhc_allele',
        'peptide': 'epitope'
    }

    # Load Experimental Data
    if tcr3d_path is not None:
        tcr3d_data = pd.read_csv(tcr3d_path, index_col=0)
        tcr3d_data.drop(['TRA', 'TRB', 'MHCseq'], axis=1, inplace=True)
        tcr3d_data.rename(columns=map_cols, inplace=True)
        tcr3d_data.reset_index(drop=True, inplace=True)
    else:
        tcr3d_data = pd.DataFrame()

    # Load AF score 3 Data
    if af_score3_path is not None:
        af_score3_data = pd.read_csv(af_score3_path)
        af_score3_data.rename(columns=map_cols, inplace=True)
        af_score3_data.reset_index(drop=True, inplace=True)
    else:
        af_score3_data = pd.DataFrame()

    
    # Load AF score 2 Data
    if af_score2_path is not None:
        af_score2_data = pd.read_csv(af_score2_path)
        af_score2_data.rename(columns=map_cols, inplace=True)
        af_score2_data.reset_index(drop=True, inplace=True)
    else:
        af_score2_data = pd.DataFrame()

    # Combine AF data
    af_data = pd.DataFrame()
    for d in [af_score3_data, af_score2_data]:
        if af_data.empty:
            af_data = d
        else:
            af_data = pd.concat([af_data, d], ignore_index=True)
    
    af_data = af_data[af_data['filepath_a'].notna() & af_data['filepath_b'].notna()]
    af_data.reset_index(drop=True, inplace=True)
    print(f'Final AlphaFold training data size:{af_data.shape}')

    if len(id10x) > 0:
        af_data = af_data[~af_data['Reference'].isin(id10x)]
        print(f'AF training data size after removing 10x references: {af_data.shape}')    
    
    # remove overlapping epitopes
    if tcr3d_path is not None:
        tcr3d_data = tcr3d_data[~tcr3d_data['epitope'].isin(af_data['epitope'])]
        print(f'TCR3D training data size after removing overlapping epitopes: {tcr3d_data.shape}')

    # Combine all data
    if tcr3d_path is None:
        train_data = af_data
    else:
        train_data = pd.concat([tcr3d_data, af_data], ignore_index=True)
    
    train_data = train_data.dropna(subset=['filepath_a', 'filepath_b'], how='any')

    print(f"Initial Positives training data size: {train_data.shape}")
    
    select_columns = ['id', 'TRA', 'TRB', 'CDR1A', 'CDR2A', 'CDR3A', 'CDR1B', 'CDR2B', 'CDR3B', 'TRA_num', 'TRB_num', 'epitope', 'MHCseq', 'mhc_allele', 'filepath_a', 'filepath_b', 'label', 'source', 'Reference']
    train_data.drop_duplicates(subset=["TRA", "TRB", "epitope", "MHCseq"], inplace=True)
    train_data = train_data[select_columns].copy()

    print(f"Positives training data size: {train_data.shape}")

    return train_data

def sort_peptides(df: pd.DataFrame) -> pd.DataFrame:
    """ Sort dataframe by peptide frequency (ascending) """
    pep_counts = df['epitope'].value_counts()
    log.info(f"Peptide counts (ascending): {pep_counts}")
    pep_order = pep_counts.sort_values(ascending=True).index.tolist()
    log.info(f"Peptide order (ascending): {pep_order}")
    return pep_order

model_params = {
    "n_output": 1,
    "dropout": 0.3,
    "n_layers": 2
}

train_params = {
    "learning_rate"   : 0.0001,
    "save_model"      : False,
    #"gamma"           : 0.5,
    "num_epochs"      : 70,
    "batch_size"      : 32,
    "pep_freq_range" : [0.005, 0.1],
    "k_top_peptides" : 5,
    "weight_decay"   : 0.01
}

config = {
    "source"          : "pdb",
    "channels"        : ["TCR", "pMHC"],
    "pairing_method"  : "basic",
    "embed_method"    : ["esm3", "atchley"],
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

#Paths
tcr3d_path = "data/02-processed/tcr3d_20251004_renamed.csv"
af_score3_path = "data/01-raw/AF_vdjdb_score3_20251212.csv"
af_score2_1217_path = "data/01-raw/AF_vdjdb_score2_wojust10x_20251217.csv"
af_score2_0105_path = "data/01-raw/AF_vdjdb_score2_wojust10x_20260105.csv"
id10x = ['34793243', '30418433', '35383307', '37872153', '32081129','30451992', '34811538']

log.info("Loading dataset:")
train_data = load_structural_data(tcr3d_path=tcr3d_path, 
                                  af_score3_path=af_score3_path).head(30)

#log.info("Loading 01/05 dataset:")
#df0105 = load_structural_data(tcr3d_path, af_score3_path, af_score2_0105_path)

#diff_df = df0105[~df0105['id'].isin(df1217['id'])]
#diff_df_wo10x = diff_df[~diff_df['Reference'].isin(id10x)]

#train_data = pd.concat([df1217, diff_df_wo10x], ignore_index=True)
#train_data = df1217.copy()
train_data.drop_duplicates(subset=["TRA", "TRB", "epitope", "MHCseq"], inplace=True)

select_columns = ['id', 'TRA', 'TRB', 'CDR1A', 'CDR2A', 'CDR3A', 'CDR1B', 'CDR2B', 'CDR3B', 'TRA_num', 'TRB_num', 'epitope', 'MHCseq', 'mhc_allele', 'filepath_a', 'filepath_b', 'label', 'source']
train_data = train_data[select_columns].copy()
train_data.to_csv("data/02-processed/tcrpMHC_combined_train_data_score3pdb.csv", index=False)
print(f"Positives training data size: {train_data.shape}")

# Create dataset
dataset = TCRpMHCDataset("data/02-processed/tcrpMHC_combined_train_data_score3pdb.csv", config=config)

# embed-specific training params

embeds = config["embed_method"]
for embed in embeds:
    if embed == "atchley":
        config["train_params"] = dict(config.get("train_params", {}), norm=True, out_channels=16) #16 in original
    elif embed == "esm3":
        config["train_params"] = dict(config.get("train_params", {}), out_channels=128)


    log.info("Generating graphs...")
    channels = ChannelsGraph(dataset, config=config, embed_method=embed)
    print(f"Channels generated for embed method '{embed}': {len(channels)}")
    
    #save channels
    torch.save(channels, f"data/02-processed/channels_graph_{embed}_score3pdb.pt")