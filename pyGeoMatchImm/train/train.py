#!/usr/bin/env python3

# local
from pyGeoMatchImm.metrics.train_metrics import (get_accuracy, 
                                                 get_auc01, get_auc, get_conf_matrix, 
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


def compute_metrics(val_results, val_idx, train_idx, peptide, best_epoch=None):
            # Compute metrics
        auc01, ap = get_auc01(val_results['labels'], val_results['predictions'])
        auc, _ = get_auc(val_results['labels'], val_results['predictions'])
        acc = get_accuracy(val_results['labels'], val_results['predictions'], 0.5)
        tn, fp, fn, tp, precision, recall = get_conf_matrix(val_results['labels'], (val_results['predictions'] >= 0.5).astype(int))

        peptide_array = np.array([peptide]*len(val_idx))

        raw_results = {'id': val_results['ids'],
                       'peptide': peptide_array,
                       'labels': val_results['labels'],
                       'predictions': val_results['predictions']}

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
                            'val_length': len(val_idx), 
                            'train_length': len(train_idx)}
       
        if best_epoch is not None:
            summary_results['best_epoch'] = best_epoch

        log.info(f'For peptide {peptide} - AUC0.1: {auc01:.4f} - AP: {ap:.4f} - ACC: {acc:.4f} - Val size: {len(val_idx)} - Train size: {len(train_idx)}')

        return raw_results, summary_results

class CrossValidator:
    def __init__(self, 
                 model_class: Any,
                 device: Any,
                 dataset: GeoDataset,
                 peptides: list,
                 configs: dict,
                 save_path: str = None):
        
                 
        self.model_class = model_class
        self.device = device

        # peptides and sample_ids lists as np.array
        self.peptides = np.asarray(peptides)
        self.configs = TrainConfigs(**configs)

        # save path for results
        cdate = datetime.datetime.now()
        self.save_dir = self.configs.save_dir if self.configs.save_dir is not None else '.'
        self.save_path = f"{self.save_dir}/results_{model_class.__name__}_{cdate.year}{cdate.month:02d}{cdate.day:02d}_summary.csv" if save_path is None else save_path
        self.channels = self.configs.channels

        self.cfg = self.configs.train_params
        self.batch_size = self.cfg.get("batch_size", 32)
        self.k_top_peptides = self.cfg.get("k_top_peptides", 5)
        self.pfreqrange = self.cfg.get("pep_freq_range", [0.1, 0.2])

        # Preprocessing dataset (normalization, etc.)
        if self.configs.train_params.get('norm', False):
            self.dataset = self.normalize_embeddings(dataset)
        else:
            self.dataset = dataset

        log.info(f'Cross-validation on top {self.k_top_peptides} peptides with frequency range {self.pfreqrange}...')
        log.info(f'Using batch size: {self.batch_size}')

    def collate_fn(self, batch):
        ch1, ch2, y = zip(*batch)
        return Batch.from_data_list(ch1), Batch.from_data_list(ch2), Batch.from_data_list(y)

    def _topk_peptides_split(self):

        peptides = np.asarray(self.peptides)
        counts = Counter(peptides.tolist())
        counts = {p: c/len(peptides) for p, c in counts.items()} # frequencies
        ranked = sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))

        top_peps = [p for p, c in ranked if self.pfreqrange[0] <= c <= self.pfreqrange[1]][:self.k_top_peptides]

        log.info(f"Peptide frequencies: {ranked[:10]} ...")  # Show top 10 peptides with frequencies
        log.info(f"Total unique peptides: {len(counts)}")
        log.info(f"Selected top {len(top_peps)} peptides for cross-validation: {top_peps}")

        if len(top_peps) == 0:
            raise ValueError(f"No peptides found in the specified frequency range {self.pfreqrange}. Adjust the range or check the dataset.")
        
        for pep in top_peps:
            test_mask = (peptides == pep)
            train_mask = (peptides != pep)

            test_idx = np.where(test_mask)[0]
            train_idx = np.where(train_mask)[0]

            yield train_idx, test_idx, pep

    def normalize_embeddings(self, dataset: torch.Tensor) -> torch.Tensor:
        """ Normalize embeddings to have zero mean and unit variance """
        scaler = StandardScaler()
        all_emb = torch.vstack([data[channel].x for data in dataset for channel in self.channels])
        scaler.fit(all_emb)
        
        for data in dataset:
            for channel in self.channels:
                data[channel].x = torch.tensor(scaler.transform(data[channel].x), dtype=data[channel].x.dtype)

        return dataset

    def run(self, save_path=None):

        self.summary_results = []
        self.raw_results = []
        self.summary_results_best = []
        self.raw_results_best = []
        self.losses = {}

        for train_idx, val_idx, pep in self._topk_peptides_split():

            log.info(f'Starting peptide {pep} with {len(val_idx)} test samples and {len(train_idx)} train samples...')
            # 1) Vectorized labels for the TRAIN partition 
            y_train = self.dataset.y_at(train_idx).view(-1).cpu().numpy()  # shape [len(train_idx)]

            # 3) Create Subsets from the main dataset
            train_logo = Subset(self.dataset, train_idx)
            val_logo   = Subset(self.dataset, val_idx)

            # 4) DataLoaders
            train_loader = GeoDataLoader(train_logo, batch_size=self.batch_size, shuffle=False)
            val_loader   = GeoDataLoader(val_logo,   batch_size=self.batch_size, shuffle=False)

            log.info(f'------------------ {pep} ------------------')
            
            # Initialize model
            node_feature_len = train_loader.dataset[0][ self.channels[0] ].x.shape[1]
            #for i in range(1, 8):
            #    print('Label:', train_loader.dataset[i]['id'])
            #    print('ID:', train_loader.dataset[i]['y'])


            log.info(f'Node feature length: {node_feature_len}')
            model = self.model_class(self.configs, node_features_len=node_feature_len)

            # Train the model (epochs loops)
            trainer = Trainer(model, self.device, train_loader, val_loader, self.configs, pep)
            losses_epochs, val_results_end, val_results_best, best_epoch = trainer.fit()

            # Store losses
            self.losses[pep] = {
                "train_loss": [le[0] for le in losses_epochs],
                "val_loss": [le[1] for le in losses_epochs]
            }

            # Compute metrics
            log.info(f"FINAL RESULTS WITH END MODEL:")
            raw_results_end, summary_results_end = compute_metrics(val_results_end, 
                                                                   val_idx, 
                                                                   train_idx,
                                                                   pep)
            
            self.raw_results.append(raw_results_end)
            self.summary_results.append(summary_results_end)

            log.info(f"FINAL RESULTS WITH BEST MODEL:")
            raw_results_best, summary_results_best = compute_metrics(val_results_best, 
                                                                     val_idx, 
                                                                     train_idx,
                                                                     pep,
                                                                     best_epoch)

            self.raw_results_best.append(raw_results_best)
            self.summary_results_best.append(summary_results_best)

        # raw results - curve analyses
        # Concatenate raw_results for all peptides
        
        self.raw_results = pd.DataFrame({
            'id': np.concatenate([r['id'] for r in self.raw_results]),
            'peptide': np.concatenate([r['peptide'] for r in self.raw_results]),
            'labels': np.concatenate([r['labels'] for r in self.raw_results]),
            'predictions': np.concatenate([r['predictions'] for r in self.raw_results])
        })

        self.raw_results_best = pd.DataFrame({
            'id': np.concatenate([r['id'] for r in self.raw_results_best]),
            'peptide': np.concatenate([r['peptide'] for r in self.raw_results_best]),
            'labels': np.concatenate([r['labels'] for r in self.raw_results_best]),
            'predictions': np.concatenate([r['predictions'] for r in self.raw_results_best])
        })

        self.raw_results.to_csv(self.save_path.replace('_summary.csv', '_raw.csv'), index=False)
        self.raw_results_best.to_csv(self.save_path.replace('_summary.csv', '_raw_best.csv'), index=False)

        # Summary Results
        self.summary_results = pd.DataFrame(self.summary_results)
        self.summary_results.to_csv(self.save_path, index=False)

        self.summary_results_best = pd.DataFrame(self.summary_results_best)
        self.summary_results_best.to_csv(self.save_path.replace('_summary.csv', '_summary_best.csv'), index=False)



class Trainer:
    def __init__(self, model, device, trainloader, valloader, 
                configs: TrainConfigs, peptide: str):
        
        self.peptide = peptide
        cfg = configs.train_params
        self.save_dir = configs.save_dir if configs.save_dir is not None else '.'
        self.n_epochs_stop = cfg.get("n_epochs_stop", 6)
        self.num_epochs = cfg.get("num_epochs", 50)
        self.weight_decay = cfg.get("weight_decay", 0.00005)
        self.ch_names = configs.channels

        self.device = device
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        #TODO : add other optimizers/losses
        self.loss_func = nn.BCEWithLogitsLoss()
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg["learning_rate"], weight_decay=self.weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"], weight_decay=self.weight_decay)
        self.scheduler = None #MultiStepLR(self.optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"])

        # early stopping
        self.epochs_no_improve = 0
        self.early_stop = False
        
        # tracking bests
        self.min_loss = float("inf")
        self.best_model = None
        self.best_accuracy = 0
        self.best_macro_auc = 0
        self.best_auc01 = 0
        self.best_auc01_result = None
        self.best_auc01_epoch = -1
        self.best_acc_epoch = -1
        self.min_loss_epoch = -1

        log.info(f'Training on {len(self.trainloader.dataset)} samples.....')
        log.info(f'Validation on {len(self.valloader.dataset)} samples.....')

    def train_one_epoch(self, epoch):

        self.model.train()
        torch.set_grad_enabled(True)
        
        predictions_tr = torch.Tensor()
        labels_tr = torch.Tensor()

        embeddings = {
            'concat_embed': [],
            'posmlp': [],
            'label': [],
            'probs': []
        }
        all_ids = []

        for count, batch in enumerate(self.trainloader):

            label = batch["y"]
            all_ids.extend(batch["id"])
            batch = batch.to(self.device)
            
            logits = self.model(batch)
            
            loss_bce = self.loss_func(logits, label.view(-1,1).float().to(self.device))
            loss = loss_bce #+ 0.01 * loss_cl 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu()
            predictions_tr = torch.cat((predictions_tr, probs), 0)
            labels_tr = torch.cat((labels_tr, label.view(-1,1).cpu()), 0)

            # visualize
            if epoch % 10 == 0 or epoch == 1 or epoch == self.num_epochs:
                embeddings['concat_embed'].extend(self.model.concat_embed)
                #embeddings['posmlp'].extend(self.model.posmlp)
                embeddings['label'].extend(label.view(-1).cpu())
                embeddings['probs'].extend(probs.view(-1).cpu())

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    print(f"{name}: no gradient!")
        if epoch % 10 == 0 or epoch == 1 or epoch == self.num_epochs:
            visualize_embeddings(embeddings['concat_embed'], y=embeddings['label'], epoch=epoch, 
                                 save_dir=self.save_dir, suffix=f'posconcat_{self.peptide}', ids = None)
            visualize_embeddings(embeddings['concat_embed'], y=embeddings['probs'], epoch=epoch, 
                                 save_dir=self.save_dir, suffix=f'posconcat_probs_{self.peptide}', ids = None)
            correlation_pred_labels(embeddings['label'], embeddings['probs'], epoch=epoch, save_dir=self.save_dir, suffix=f'{self.peptide}')
        

        log.info(f"Logit mean: {logits.mean().item()}, std: {logits.std().item()}")
        #self.scheduler.step()
        labels_tr = labels_tr.detach().numpy()
        predictions_tr = predictions_tr.detach().numpy()
        acc_tr = get_accuracy(labels_tr, predictions_tr, 0.5)
        
        if epoch % 10 == 0 or epoch == 1 or epoch == self.num_epochs:
            log.info(f'Epoch {epoch} / {self.num_epochs} - train_loss: {loss:.4f} - train_accuracy: {acc_tr:.4f}')
        
        return loss.item(), acc_tr

    def evaluate(self, epoch):
        
        self.model.eval()
        predictions, labels = torch.Tensor(), torch.Tensor()
        val_loss = 0.0
        all_ids = []
        with torch.no_grad():
            for batch in self.valloader:
                label = batch["y"]
                all_ids.extend(batch["id"])
                batch = batch.to(self.device)
                logits = self.model(batch)

                loss_bce = self.loss_func(logits, label.view(-1,1).float().to(self.device))
                val_loss += loss_bce.item() #+ 0.01 * loss_cl.item()

                probs = torch.sigmoid(logits).detach().cpu()
                predictions = torch.cat((predictions, probs), 0)
                labels = torch.cat((labels, label.view(-1,1).cpu()), 0)


        labels, predictions = labels.numpy().flatten(), predictions.numpy().flatten()
        
        # Calculate Metrics for validation epoch
        val_loss /= len(self.valloader)
        accuracy = get_accuracy(labels, predictions, 0.5)
        #auc_per_peptide, macro_auc = get_auc_per_peptide(labels, predictions, self.peptides)
        auc01, ap = get_auc01(labels, predictions)

        pred_results = {
            'ids': all_ids,
            'labels': labels,
            'predictions': predictions
        }
        if epoch % 10 == 0 or epoch == 1 or epoch == self.num_epochs:
            log.info(f'Epoch {epoch} / {self.num_epochs} - val_loss: {val_loss:.4f} - val_acc: {accuracy:.4f} - val_avg_precision: {ap:.4f} - val_auc01: {auc01:.4f}')
        
        return val_loss, accuracy, auc01, pred_results

    def fit(self, save_path=None):
        
        loss_epochs = []

        for epoch in range(1, self.num_epochs+1):

            train_loss, train_accuracy = self.train_one_epoch(epoch)
            val_loss, val_accuracy, val_auc01, val_results = self.evaluate(epoch)

            loss_epochs.append((train_loss, val_loss))

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_acc_epoch = epoch

            # save the best model based on AUC
            if val_auc01 > self.best_auc01:
                self.best_auc01 = val_auc01
                self.best_auc01_epoch = epoch
                self.best_auc01_result = val_results
                self.best_model = self.model
                if save_path is not None:
                    torch.save(self.model.state_dict(), save_path)
                    log.info("Model saved (best AUC0.1).")
                    self.best_model = self.model

            # early stopping based on validation loss
            if val_loss < self.min_loss:
                self.epochs_no_improve = 0
                self.min_loss = val_loss
                self.min_loss_epoch = epoch
            else:
                self.epochs_no_improve += 1

            #if epoch > 5 and self.epochs_no_improve == self.n_epochs_stop:
            #    log.info("Early stopping triggered!")
            #    self.early_stop = True
            #    break

        log.info(f"min_val_loss: {self.min_loss:.4f} at epoch {self.min_loss_epoch} "
                 f"--- best_val_acc: {self.best_accuracy:.4f} at epoch {self.best_acc_epoch}"
                 f"--- best_val_auc01: {self.best_auc01:.4f} at epoch {self.best_auc01_epoch}")

        return loss_epochs, val_results, self.best_auc01_result, self.best_auc01_epoch

