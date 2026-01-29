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

#Data
train_data_path = "data/02-processed/tcrpMHC_combined_train_data_1217.csv"
train_data = pd.read_csv(train_data_path)
channels_graph_path = "data/02-processed/channels_graph_1217.pt"
channels = torch.load(channels_graph_path, weights_only=False)

model_params = {
    "n_output": 1,
    "dropout": 0.3,
    "n_layers": 2
}

train_params = {
    "learning_rate"   : 0.0001,
    "save_model"      : False,
    #"gamma"           : 0.5,
    "weighted_loss"   : False,
    "loss_function"    : "BCEWithLogits",
    "num_epochs"      : 70,
    "batch_size"      : 16,
    "pep_freq_range" : [0.005, 0.1],
    "k_top_peptides" : 5,
    "weight_decay"   : 0.01
}

config = {
    "source"          : "pdb",
    "channels"        : ["TCR", "pMHC"],
    "pairing_method"  : "basic",
    "embed_method"    : ["atchley"],
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

# Define models to train
models_dict = {
"GIN_cn1_ce1": {'model_class': XATTGraph,
                            'graph_encoder': 'GIN',
                            'cross_embed': True,
                            'cross_nodes': True
                            }
}

# embed-specific training params
embed = config["embed_method"][0]
if embed == "atchley":
    config["train_params"] = dict(config.get("train_params", {}), norm=True, out_channels=16) #16 in original
elif embed == "esm3":
    config["train_params"] = dict(config.get("train_params", {}), norm=False, out_channels=128)
    
chids, chlabels, chpeptides, chtras, chtrbs = zip(*[(i.get('id'), i.get('label'), i.get('peptide'), i.get('TRA'), i.get('TRB')) for i in channels])

ds = ChannelsPairDataset(
    channels_graph=channels,
    ch1_name=channels.channel_names[0],
    ch2_name=channels.channel_names[1],
)

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
        config.update({
            "cross_nodes"   : cross_nodes,
            "cross_embed"   : cross_embed
            })
        
    log.info("Running architecture: %s", arch)

    # save dir per-run
    dropout = int(config["model_params"].get("dropout", 0) * 10)
    save_dir = f"developments/weighted_training/{arch}_{embed}_drop{dropout}"
    config["save_dir"] = save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    basename = save_dir.split("/")[-1]

    try:    
        cv = CrossValidator(
            model_class=model_class,
            dataset=ds,
            device=device,
            peptides=chpeptides,
            configs=config,

        )

        cv.run()

        with open(os.path.join(save_dir, "config.json"), "w") as fp:
            json.dump(config, fp, indent=4)

        plot_precision_recall_curve(cv.raw_results['labels'],
                        cv.raw_results['predictions'],
                        save_dir)

        plot_roc_curve(cv.raw_results['labels'],
                cv.raw_results['predictions'],
                save_dir)

        plot_loss_curve_logocv(cv.losses, save_dir=save_dir)

    except Exception:
        log.exception("Run failed for %s with embed=%s", arch, embed)

    del cv, ds, channels
    gc.collect()
    torch.cuda.empty_cache()

    if __name__ == "__main__":
        pass

