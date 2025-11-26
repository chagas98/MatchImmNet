#!/usr/bin/env python3

from pyGeoMatchImm.data.datasets import TCRpMHCDataset, ChannelsPairDataset, ChannelsGraph
from pyGeoMatchImm.train.train import CrossValidator
from pyGeoMatchImm.models.Jha import Jha_GCN, Jha_GAT
from pyGeoMatchImm.models.models import (CrossAttentionGCN, 
                                         CrossAttentionGAT, 
                                         CrossAttentionGIN,
                                         CrossAttentionGATGIN, 
                                            MultiGCN,   
                                            MultiGAT,
                                            MultiGIN) 
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
import os
import json
import logging as log


# logger
log.basicConfig(level=log.INFO)
log.getLogger("pyGeoMatchImm").setLevel(log.INFO)
log.getLogger("MDAnalysis").setLevel(log.WARNING)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

map_cols = {
    'TCR_ID': 'id',
    'TRA_ref': 'TRA',
    'TRB_ref': 'TRB',
    'MHCseq_ref': 'MHCseq',
    'assigned_allele': 'mhc_allele',
    'peptide': 'epitope'
}

tcr3d_data_path = "data/02-processed/tcr3d_20251004_renamed.csv"
tcr3d_data = pd.read_csv(tcr3d_data_path, index_col=0)
tcr3d_data.drop(['TRA', 'TRB', 'MHCseq'], axis=1, inplace=True)
tcr3d_data.rename(columns=map_cols, inplace=True)
tcr3d_data.reset_index(drop=True, inplace=True)
print(tcr3d_data.head())
af_data_path = "data/02-processed/AF_vdjdb_score3_20251026.csv"
af_data = pd.read_csv(af_data_path)
af_data.rename(columns=map_cols, inplace=True)
af_data.reset_index(drop=True, inplace=True)
print(af_data.head())

# remove overlapping epitopes
tcr3d_data = tcr3d_data[~tcr3d_data['epitope'].isin(af_data['epitope'])]

train_data = pd.concat([tcr3d_data, af_data], ignore_index=True)
train_data = train_data.dropna(subset=['filepath_a', 'filepath_b'], how='any')


select_columns = ['id', 'TRA', 'TRB', 'CDR1A', 'CDR2A', 'CDR3A', 'CDR1B', 'CDR2B', 'CDR3B', 'TRA_num', 'TRB_num', 'epitope', 'MHCseq', 'mhc_allele', 'filepath_a', 'filepath_b', 'label', 'source']
train_data.drop_duplicates(subset=["TRA", "TRB", "epitope", "MHCseq"], inplace=True)
train_data = train_data[select_columns].copy()
train_data.to_csv("tcrpMHC_combined_train_data.csv", index=False)


model_params = {
    "n_output": 1,
    "dropout": 0.3,
    "n_layers": 2
}

train_params = {
    "learning_rate"   : 0.0001,
    #"milestones"      : [1, 5],
    #"gamma"           : 0.5,
    "num_epochs"      : 60,
    "batch_size"      : 16,
    "pep_freq_range" : [0.005, 0.1],
    "k_top_peptides" : 10,
    "weight_decay"   : 0.01
}

config = {
    "source"          : "pdb",
    "channels"        : ["TCR", "pMHC"],
    "pairing_method"  : "basic",
    "embed_method"    : ["esm3", "atchley"],
    "graph_method"    : "graphein",
    "negative_prop"   : 5,
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

dataset = TCRpMHCDataset("tcrpMHC_combined_train_data.csv", config=config)


models_dict = {
    "jha_gcn": Jha_GCN,
    "jha_gat": Jha_GAT,
    "cagcn": CrossAttentionGCN,
    "cagat": CrossAttentionGAT,
    "cagin": CrossAttentionGIN,
    "cagatgin": CrossAttentionGATGIN,
    "multigcn": MultiGCN,
    "multigat": MultiGAT,
    "multigin": MultiGIN
}

for j, embed in enumerate(config['embed_method']):
    for i, arch in enumerate([ "cagin", "multigin", "cagat", "multigat", "cagcn", "multigcn"]):

        if embed == 'esm3' and arch in ["cagin", "multigin"]:
            log.info(f"Skipping architecture: {arch} with embedding method: {embed}")
            continue

        model_class = models_dict[arch]

        log.info(f"Running architecture: {arch}")
        log.info(f"Using embedding method: {embed}")

        if embed == "atchley":
            config["train_params"]['norm'] = True
            config["train_params"]['out_channels'] = 16
        else:
            config["train_params"]['norm'] = False
            config["train_params"]['out_channels'] = 128


        save_dir = f"./{arch}_neg{config['negative_prop']}_{embed}_dp0{int(config['model_params']['dropout']*10)}"
        config["save_dir"] = save_dir

        channels = ChannelsGraph(dataset, config=config, embed_method=embed)
        peptides = channels.get_seq_chain(name="pMHC", chain="C")


        ds = ChannelsPairDataset(
                ids=channels.ids,
                ch1_graphs=channels.ch1,
                ch2_graphs=channels.ch2,
                labels=channels.get_labels(),   # or channels.y_tensor
                ch1_name=channels.channel_names[0],
                ch2_name=channels.channel_names[1],
            )

        validate_data(train_data, dict_lists={
            "id": channels.ids,
            "epitope": peptides,
            "label": channels.get_labels(),
            "TRA": channels.get_seq_chain(name="TCR", chain="D"),
            "TRB": channels.get_seq_chain(name="TCR", chain="E")
        },
            cols_to_check=["epitope", "label", "TRA", "TRB"])
        
        cv = CrossValidator(model_class=model_class,
                            dataset=ds, 
                            device=device,
                            peptides=peptides,
                            configs=config)

        cv.run()


        json.dump(config, open(f"./{save_dir}/config.json", "w"), indent=4)

        plot_precision_recall_curve(cv.raw_results['labels'], 
                                    cv.raw_results['predictions'], 
                                    save_dir)

        plot_roc_curve(cv.raw_results['labels'],
                        cv.raw_results['predictions'],
                        save_dir)

        plot_loss_curve_logocv(cv.losses, save_dir=save_dir)

        # reset pytorch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # clean cache, memory release
        del channels
        del ds
        del cv
        


if __name__ == "__main__":
    pass