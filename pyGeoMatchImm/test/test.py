#!/usr/bin/env python3

# local
from pyGeoMatchImm.metrics.train_metrics import (get_accuracy, 
                                                 get_auc01,
                                                 get_auc,
                                                 get_conf_matrix, 
                                                 plot_precision_recall_curve,
                                                 plot_roc_curve,
                                                 plot_loss_curve_logocv)
from sklearn.preprocessing import StandardScaler
from ..utils.base import TrainConfigs
from ..utils.visualize import visualize_embeddings, correlation_pred_labels
# third-parties
import math
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import get_graph_node_names
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Data,  Batch
from torch_geometric.data import Dataset as GeoDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any
from collections import Counter
import datetime
import numpy as np
import logging
log = logging.getLogger(__name__)

torch.manual_seed(42); np.random.seed(42)


def compute_metrics_per_peptide(outputs):
            # Compute metrics
        peptides = outputs['peptides']
        final_results = []
        seen=set()
        for peptide in np.unique(peptides):
            if peptide in seen:
                continue
            seen.add(peptide)

            mask = (peptides == peptide)
            mask_size = mask.sum()

            y_true_pep = outputs['labels'][mask]
            y_pred_pep = outputs['predictions'][mask]

            if len(np.unique(y_true_pep)) < 2:
                continue
            
            auc01, ap = get_auc01(y_true_pep, y_pred_pep)
            auc, _ = get_auc(y_true_pep, y_pred_pep)
            acc = get_accuracy(y_true_pep, y_pred_pep, 0.5)
            tn, fp, fn, tp, precision, recall = get_conf_matrix(y_true_pep, (y_pred_pep >= 0.5).astype(int))
    
            
            summary_results = {'peptide': peptide,
                                'auc': auc,
                                'auc01': auc01,
                                'ap': ap,
                                'acc': acc,
                                'tn': tn,
                                'fp': fp,
                                'fn': fn,
                                'tp': tp,
                                'precision': precision,
                                'recall': recall, 
                                'size': mask_size}

            log.info(f'For peptide {peptide} - AUC0.1: {auc01:.4f} - AP: {ap:.4f} - ACC: {acc:.4f} - Size: {mask_size}')

            final_results.append(summary_results)

        out_df = pd.DataFrame(final_results)
        return out_df

class Tester:
    def __init__(self, 
                model_class: Any, 
                device: Any, 
                configs: dict):

        self.model_class = model_class
        self.device = device
        self.configs = TrainConfigs(**configs)
        self.save_dir = self.configs.save_dir if self.configs.save_dir is not None else './'
        self.train_cfg = self.configs.train_params
        self.batch_size = self.train_cfg.get("batch_size", 32)


    def normalize_embeddings(self, dataset: torch.Tensor) -> torch.Tensor:
        """ Normalize embeddings to have zero mean and unit variance """
        scaler = StandardScaler()
        all_emb = torch.vstack([data[channel].x for data in dataset for channel in self.channels])
        scaler.fit(all_emb)
        
        for data in dataset:
            for channel in self.channels:
                data[channel].x = torch.tensor(scaler.transform(data[channel].x), dtype=data[channel].x.dtype)

        return dataset
    def predict(self, data):

        # Preprocessing dataset (normalization, etc.)
        if self.train_cfg.get('norm', False):
            data = self.normalize_embeddings(data)

        log.info("Starting predictions on test set...")

        test_loader = GeoDataLoader(data, batch_size=self.batch_size, shuffle=False)
        node_feature_len = test_loader.dataset[0][self.channels[0]].x.shape[1]

        log.info(f"Node feature length: {node_feature_len}")

        # Inference
        self.model.eval()
        predictions, labels = torch.Tensor(), torch.Tensor()
        val_loss = 0.0
        all_ids = []
        all_peptides = []
        with torch.no_grad():
            for batch in test_loader:
                label = batch['y']
                all_ids.extend(batch['id'])
                peptides = batch['peptide']
                all_peptides.extend(peptides)

                # Models
                inputs = batch.to(self.device)
                self.model = self.model_class(self.configs, self.node_feature_len).to(self.device)
                logits = self.model(inputs)
                
                # calculate loss
                loss_bce = self.loss_func(logits, label.view(-1,1).float().to(self.device))
                test_loss += loss_bce.item() 

                # calculate metrics
                probs = torch.sigmoid(logits).detach().cpu()
                predictions = torch.cat((predictions, probs), 0)
                labels = torch.cat((labels, label.view(-1,1).cpu()), 0)

        labels, predictions = labels.numpy().flatten(), predictions.numpy().flatten()

        pred_results = {
            'ids': all_ids,
            'peptides': all_peptides,
            'labels': labels,
            'predictions': predictions
        }

        processed_results = compute_metrics_per_peptide(pred_results)
        pd.DataFrame(pred_results).to_csv(f"{self.save_dir}/raw_test_results.csv", index=False)
        pd.DataFrame(processed_results).to_csv(f"{self.save_dir}/summary_test_results.csv", index=False)

        return processed_results