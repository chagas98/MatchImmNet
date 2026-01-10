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

# logger
log.basicConfig(level=log.INFO)
log.getLogger("pyGeoMatchImm").setLevel(log.INFO)
log.getLogger("MDAnalysis").setLevel(log.WARNING)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data Paths
tcr3d_path = "data/02-processed/tcr3d_20251004_renamed.csv"
af_score3_path = "data/01-raw/AF_vdjdb_score3_20251212.csv"
af_score2_path = "data/01-raw/AF_vdjdb_score2_wojust10x_20260105.csv"
filter_10x = True
pmid10x = ['34793243', '30418433', '35383307', '37872153', '32081129','30451992', '34811538']

map_cols = {
    'TCR_ID': 'id',
    'TRA_ref': 'TRA',
    'TRB_ref': 'TRB',
    'MHCseq_ref': 'MHCseq',
    'assigned_allele': 'mhc_allele',
    'peptide': 'epitope'
}

# Load Experimental Data
tcr3d_data = pd.read_csv(tcr3d_path, index_col=0)
tcr3d_data.drop(['TRA', 'TRB', 'MHCseq'], axis=1, inplace=True)
tcr3d_data.rename(columns=map_cols, inplace=True)
tcr3d_data.reset_index(drop=True, inplace=True)

# Load AF score 3 Data
af_score3_data = pd.read_csv(af_score3_path)
af_score3_data.rename(columns=map_cols, inplace=True)
af_score3_data.reset_index(drop=True, inplace=True)

# Load AF score 2 Data
af_score2_data = pd.read_csv(af_score2_path)
af_score2_data.rename(columns=map_cols, inplace=True)
af_score2_data.reset_index(drop=True, inplace=True)

af_data = pd.concat([af_score3_data, af_score2_data], ignore_index=True)
af_data = af_data[af_data['filepath_a'].notna() & af_data['filepath_b'].notna()]
af_data.reset_index(drop=True, inplace=True)

if len(pmid10x) > 0 and filter_10x:
    af_data = af_data[~af_data['Reference'].isin(pmid10x)]
    print(f'Removed 10x samples from AF data: {pmid10x}')

else:
    print(f'No 10x samples removed from AF data.')

print(f'Final AlphaFold training data size:{af_data.shape}')

# remove overlapping epitopes
tcr3d_data = tcr3d_data[~tcr3d_data['epitope'].isin(af_data['epitope'])]

print(f'TCR3D training data size after removing overlapping epitopes: {tcr3d_data.shape}')

train_data = pd.concat([tcr3d_data, af_data], ignore_index=True)
train_data = train_data.dropna(subset=['filepath_a', 'filepath_b'], how='any')

select_columns = ['id', 'TRA', 'TRB', 'CDR1A', 'CDR2A', 'CDR3A', 'CDR1B', 'CDR2B', 'CDR3B', 'TRA_num', 'TRB_num', 'epitope', 'MHCseq', 'mhc_allele', 'filepath_a', 'filepath_b', 'label', 'source']
train_data.drop_duplicates(subset=["TRA", "TRB", "epitope", "MHCseq"], inplace=True)
train_data = train_data[select_columns].copy()
train_data.to_csv("data/02-processed/tcrpMHC_combined_train_data.csv", index=False)
print(f"Positives training data size: {train_data.shape}")

model_params = {
    "n_output": 1,
    "dropout": 0.3,
    "n_layers": 2
}

train_params = {
    "learning_rate"   : 0.0001,
    "save_model"      : False,
    #"gamma"           : 0.5,
    "num_epochs"      : 100,
    "batch_size"      : 16,
    "pep_freq_range" : [0.005, 0.1],
    "k_top_peptides" : 3,
    "weight_decay"   : 0.01
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

dataset = TCRpMHCDataset("data/02-processed/tcrpMHC_combined_train_data.csv", config=config)

# Define models to train
models_dict = {
    "cangin": {'model_class': CrossAttentionNodesGIN}
}

# add architectures with cross attention variants
for graph_enc, cross_nodes, cross_embed in product(
    ["GIN"], [True, False], [True, False]):
    arch = f"{graph_enc}_cn{int(cross_nodes)}_ce{int(cross_embed)}"

    models_dict.update({arch: {'model_class': XATTGraph,
                                'cross_embed': cross_embed,
                                'cross_nodes': cross_nodes
                                }})

print(f"Starting for loop over architectures and embedding methods...")
for embed in config['embed_method']:

    # prepare run config for each architecture
    run_cfg = deepcopy(config)
        
    log.info("Using embedding method: %s", embed)

    # embed-specific training params
    if embed == "atchley":
        run_cfg["train_params"] = dict(run_cfg.get("train_params", {}), norm=True, out_channels=16) #16 in original
    else:
        run_cfg["train_params"] = dict(run_cfg.get("train_params", {}), norm=False, out_channels=128) #128 in original


    log.info("Generating graphs...")
    channels = ChannelsGraph(dataset, config=run_cfg, embed_method=embed)
    peptides = [ch.get('peptide') for ch in channels]

    ds = ChannelsPairDataset(
        channels_graph=channels,
        ch1_name=channels.channel_names[0],
        ch2_name=channels.channel_names[1],
    )

    chids, chlabels, chpeptides, chtras, chtrbs = zip(*[(i.get('id'), i.get('label'), i.get('peptide'), i.get('TRA'), i.get('TRB')) for i in channels])
    validate_data(train_data, dict_lists={
        "id": chids,
        "epitope": chpeptides,
        "label": chlabels,
        "TRA": chtras,
        "TRB": chtrbs
    }, cols_to_check=["epitope", "label", "TRA", "TRB"])

    for arch, model_cfg in models_dict.items():
        
        model_class = model_cfg['model_class']

        if model_class == XATTGraph:
            cross_embed = model_cfg['cross_embed']
            cross_nodes = model_cfg['cross_nodes']
            run_cfg.update({
                "graph_encoder" : graph_enc,
                "cross_nodes"   : cross_nodes,
                "cross_embed"   : cross_embed
                })
            
        log.info("Running architecture: %s", arch)

        # save dir per-run
        dropout = int(run_cfg["model_params"].get("dropout", 0) * 10)
        save_dir = f"developments/increase_dataset_0105_neg{run_cfg['negative_prop']}_bs{train_params['batch_size']}_lr{train_params['learning_rate']*10000}_no10x/{arch}_neg{run_cfg['negative_prop']}_{embed}_dp0{dropout}"
        run_cfg["save_dir"] = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        basename = save_dir.split("/")[-1]
        
        #if basename in ["GIN_cn1_ce1_neg3_esm3_dp03"]:
        #    log.info(f"Skipping architecture: {arch} with embedding method: {embed}")
        #    continue

        try:    
            cv = CrossValidator(
                model_class=model_class,
                dataset=ds,
                device=device,
                peptides=peptides,
                configs=run_cfg
            )

            cv.run()

            with open(os.path.join(save_dir, "config.json"), "w") as fp:
                json.dump(run_cfg, fp, indent=4)

            plot_precision_recall_curve(cv.raw_results['labels'],
                            cv.raw_results['predictions'],
                            save_dir)

            plot_roc_curve(cv.raw_results['labels'],
                    cv.raw_results['predictions'],
                    save_dir)

            plot_loss_curve_logocv(cv.losses, save_dir=save_dir)

        except Exception:
            log.exception("Run failed for %s with embed=%s", arch, embed)
        
        finally:
            # reset pytorch & free memory
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

            for obj in ("cv"):
                if obj in locals():
                    try:
                        del globals()[obj]
                    except Exception:
                        try:
                            del locals()[obj]
                        except Exception:
                            pass
                


if __name__ == "__main__":
    pass