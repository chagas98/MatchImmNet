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


def load_structural_data(tcr3d_path: str,
                         af_score3_path: str,
                         af_score2_path: str,
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

    if len(id10x) > 0:
        af_data = af_data[~af_data['Reference'].isin(id10x)]

    print(f'Final AlphaFold training data size:{af_data.shape}')

    # remove overlapping epitopes
    tcr3d_data = tcr3d_data[~tcr3d_data['epitope'].isin(af_data['epitope'])]

    print(f'TCR3D training data size after removing overlapping epitopes: {tcr3d_data.shape}')

    train_data = pd.concat([tcr3d_data, af_data], ignore_index=True)
    train_data = train_data.dropna(subset=['filepath_a', 'filepath_b'], how='any')

    print(f'Total training data size: {train_data.shape}')

    select_columns = ['id', 'TRA', 'TRB', 'CDR1A', 'CDR2A', 'CDR3A', 'CDR1B', 'CDR2B', 'CDR3B', 'TRA_num', 'TRB_num', 'epitope', 'MHCseq', 'mhc_allele', 'filepath_a', 'filepath_b', 'label', 'source', 'Reference']
    train_data.drop_duplicates(subset=["TRA", "TRB", "epitope", "MHCseq"], inplace=True)
    train_data = train_data[select_columns].copy()

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

#Paths
tcr3d_path = "data/02-processed/tcr3d_20251004_renamed.csv"
af_score3_path = "data/01-raw/AF_vdjdb_score3_20251212.csv"
af_score2_1217_path = "data/01-raw/AF_vdjdb_score2_wojust10x_20251217.csv"
af_score2_0105_path = "data/01-raw/AF_vdjdb_score2_wojust10x_20260105.csv"
id10x = ['34793243', '30418433', '35383307', '37872153', '32081129','30451992', '34811538']

log.info("Loading 12/17 dataset:")
df1217 = load_structural_data(tcr3d_path, af_score3_path, af_score2_1217_path)

log.info("Loading 01/05 dataset:")
df0105 = load_structural_data(tcr3d_path, af_score3_path, af_score2_0105_path)

diff_df = df0105[~df0105['id'].isin(df1217['id'])]
diff_df_wo10x = diff_df[~diff_df['Reference'].isin(id10x)]

add_peptides = sort_peptides(diff_df_wo10x)

log.info(f"Additional peptides in 01/05 dataset: {add_peptides}")

# cumulative addition of peptides
train_data = df1217.copy()
#already_added_peps = ['RMYSPVSIL', 'KIYSKHTPI', 'RARSVASQSI', 'LLFGYPVYV', 'SLLMWITQC', 'EPLPQGQLTAY', 'EENLLDFVRF', 'EEYLKAWTF', 'EEYLQAFTY', 'GLCTLVAML', 'ALWGPDPAAA', 'RFPLTFGWCF', 'SMLGIGIVPV', 'MMWDRGLGMM']
already_added_peps = ['RMYSPVSIL', 'KIYSKHTPI', 'RARSVASQSI', 'LLFGYPVYV', 'SLLMWITQC', 'EPLPQGQLTAY', 'EENLLDFVRF', 'EEYLKAWTF', 'EEYLQAFTY', 'GLCTLVAML', 'ALWGPDPAAA', 'RFPLTFGWCF', 'SMLGIGIVPV', 'MMWDRGLGMM', 'SLYNTIATL', 'LPFDKPTIM', 'LPFEKSTVM', 'LPFERATIM', 'LRVMMLAPF', 'GQVMVVAPR', 'YLEPGPVTA', 'GPEPLPQGQLTAY', 'ALQIPFAMQM', 'RMYSPVSI', 'YPDKVFRSSV', 'FPQSAPHGVVF', 'KLVALVINAV', 'YQFGPDFPIA', 'RQWGPDPAAV', 'YLGGPDFPTI', 'MVWGPDPLYV', 'RQFGPDFPTI', 'TAFTIPSI', 'RQFGPDWIVA', 'AQWGPDPAAA', 'ILAKFLHWL', 'SLYNTVATL', 'SLFNTIAVL', 'VMAPRTLVL', 'GTSGSPIVNR', 'GTSGSPIINR', 'YLEPGPVTV', 'KRWIIMGLNK', 'KRWIILGLNK', 'ALGIGILTV', 'KQIYKTPPI', 'TLATHGLAAV', 'MQLFFSYFA', 'HSKKKCDEL', 'LPEPLPQGQLGAY', 'RLPAKAPL', 'LPFEKSTIM', 'RLQSLQTYV', 'FEDLRVLSF', 'GVYDGREHTV', 'LPEPLPQGGLTAY', 'LPEPLPQAQLTAY', 'IMDQVPFSV', 'GADGVGKSAL', 'LPEPLGQGQLTAY', 'RPPIFIRRL', 'HMTEVVRHC', 'LPEGLPQGQLTAY', 'RLPAKAPLLGCG', 'RLPAKAPLL', 'VPLTEDAEL', 'ILDQVPFSV', 'TPQDLNTML', 'LPFDKTTIM', 'EAAGIGILTV', 'TRLALIAPK', 'ALHGGWTTK', 'LPFDKSTIM', 'KLVALGINAV', 'ITDQVPFSV', 'LPFDKSTVM', 'LPFDKATIM', 'APRGPHGGAASGL', 'VMAPRTLIL', 'LPEPLPQGQLTAY', 'LPPAYTNSF', 'IPLTEEAEL', 'SLLMWITQV', 'FLRGRAYGL', 'MLAKALRKV', 'FAMQMAYRF', 'AAGIGILTV', 'ELAGIGILTV', 'RLITGRLQSL', 'GILGFVFTL', 'RPIIRPATL', 'QMMVKAGL', 'QYIKWPWYI', 'MEVTPSGTWL', 'YLQPRTFLL', 'KTFPPTEPK', 'TTDPSFLGRY']
group_pep = []
for i, add_pep in enumerate(add_peptides):

    if add_pep in already_added_peps:
        continue
    else:
        already_added_peps.append(add_pep)
        log.info(f"Adding peptide {i}/{len(add_peptides)}: {add_pep}")
        log.info(f"Peptides added so far: {already_added_peps}\n")

        if i % 4 != 0 and i != len(add_peptides) -1:
            group_pep.append(add_pep)
            log.info(f"Skipping training with only {add_pep} addition for brevity.")
            continue
        else:
            group_pep.append(add_pep)
            log.info(f"Proceeding with training including more these peptides: {group_pep}")
            #reset 
            group_pep = []

    pep_df = diff_df_wo10x[diff_df_wo10x['epitope'].isin(already_added_peps)]
    train_data = pd.concat([train_data, pep_df], ignore_index=True)

    log.info(f"Before dropping duplicates, training data size with peptide {add_pep}: {train_data.shape}")
    train_data.drop_duplicates(subset=["TRA", "TRB", "epitope", "MHCseq"], inplace=True)
    log.info(f"Training data size with peptide {add_pep}: {train_data.shape}")

    #drop reference column
    train_data.drop(['Reference'], axis=1, inplace=True)
    train_data.to_csv("data/02-processed/tcrpMHC_filling_combined_train_data.csv", index=False)
    
    # Create dataset
    dataset = TCRpMHCDataset("data/02-processed/tcrpMHC_filling_combined_train_data.csv", config=config)

    # Define models to train
    models_dict = {
    "GIN_cn1_ce1": {'model_class': XATTGraph,
                                'cross_embed': True,
                                'cross_nodes': True
                                }
    }

    # embed-specific training params
    embed = config["embed_method"][0]
    if embed == "atchley":
        config["train_params"] = dict(config.get("train_params", {}), norm=True, out_channels=16) #16 in original


    log.info("Generating graphs...")
    channels = ChannelsGraph(dataset, config=config, embed_method=embed)
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

        log.info("Running architecture: %s", arch)

        # save dir per-run
        dropout = int(config["model_params"].get("dropout", 0) * 10)
        save_dir = f"developments/filling_0105data/{i}_model_{add_pep}_{arch}_neg{config['negative_prop']}_{embed}_dp0{dropout}"
        config["save_dir"] = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        basename = save_dir.split("/")[-1]

        try:    
            cv = CrossValidator(
                model_class=model_class,
                dataset=ds,
                device=device,
                peptides=peptides,
                configs=config
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
        

    del cv, ds, channels, dataset
    gc.collect()
    torch.cuda.empty_cache()

    if __name__ == "__main__":
        pass



