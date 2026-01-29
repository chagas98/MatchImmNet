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

model_params = {
    "n_output": 1,
    "dropout": 0.3,
    "n_layers": 2
}

train_params = {
    "learning_rate"   : 0.0001,
    "save_model"      : False,
    "weighted_loss"   : False,
    "loss_function"   : "BCEWithLogits",
    "num_epochs"      : 70,
    "batch_size"      : 16,
    "pep_freq_range" : [0.005, 0.1],
    "k_top_peptides" : 10,
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

embed = config.get("embed_method")[0]  # "atchley" or "esm3"

#Data
train_data_path = "data/02-processed/tcrpMHC_combined_train_data_score3pdb.csv"
train_data = pd.read_csv(train_data_path)
channels_graph_path = f"data/02-processed/channels_graph_{embed}_score3pdb.pt"
channels = torch.load(channels_graph_path, weights_only=False)

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


booleanopt = [True, False]
models_dict = {}
# add architectures with cross attention variants
for graph_enc, cross_nodes, cross_embed, pe_tcr, pe_epitope, pe_mhc in product(
    ["GIN"], booleanopt, booleanopt, booleanopt, booleanopt, booleanopt):
    arch = f"{graph_enc}_cn{int(cross_nodes)}_ce{int(cross_embed)}_petcr{int(pe_tcr)}_peepi{int(pe_epitope)}_pemhc{int(pe_mhc)}_{embed}"

    models_dict.update({arch: {'model_class': XATTGraph,
                               'graph_encoder': graph_enc,
                               'cross_embed': cross_embed,
                               'cross_nodes': cross_nodes,
                                'pe_tcr': pe_tcr,
                                'pe_epitope': pe_epitope,
                                'pe_mhc': pe_mhc
                               }})

for arch, model_cfg in models_dict.items():
    run_config = deepcopy(config)
    model_class = model_cfg.get('model_class', None)
    run_config.get("model_params", {})["graph_encoder"] = model_cfg.get('graph_encoder', None)
    run_config.get("model_params", {})["cross_embed"] = model_cfg.get('cross_embed', None)
    run_config.get("model_params", {})["cross_nodes"] = model_cfg.get('cross_nodes', None)
    run_config.get("model_params", {})["pe_tcr"] = model_cfg.get('pe_tcr', None)
    run_config.get("model_params", {})["pe_epitope"] = model_cfg.get('pe_epitope', None)
    run_config.get("model_params", {})["pe_mhc"] = model_cfg.get('pe_mhc', None)

    # embed-specific training params
    if embed == "atchley":
        run_config["train_params"] = dict(run_config.get("train_params", {}), norm=True, out_channels=16) #16 in original
    elif embed == "esm3":
        run_config["train_params"] = dict(run_config.get("train_params", {}), norm=False, out_channels=128)
        
    log.info("Configuration model params: %s", run_config["model_params"])
    log.info("Running architecture: %s", arch)

    # save dir per-run
    dropout = int(run_config["model_params"].get("dropout", 0) * 10)
    save_dir = f"developments/score3/{arch}_{embed}_drop{dropout}"
    run_config["save_dir"] = save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    basename = save_dir.split("/")[-1]

    try:    
        cv = CrossValidator(
            model_class=model_class,
            dataset=ds,
            device=device,
            peptides=chpeptides,
            configs=run_config
        )

        cv.run()

        with open(os.path.join(save_dir, "config.json"), "w") as fp:
            json.dump(run_config, fp, indent=4)

        plot_precision_recall_curve(cv.raw_results['labels'],
                        cv.raw_results['predictions'],
                        save_dir)

        plot_roc_curve(cv.raw_results['labels'],
                cv.raw_results['predictions'],
                save_dir)

        plot_loss_curve_logocv(cv.losses, save_dir=save_dir)

    except Exception:
        log.exception("Run failed for %s with embed=%s", arch, embed)

    del cv
    gc.collect()
    torch.cuda.empty_cache()

    if __name__ == "__main__":
        pass

