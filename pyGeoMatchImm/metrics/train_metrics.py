#!/usr/bin/env python3

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

import logging

log = logging.getLogger(__name__)

def get_accuracy(y_true, y_pred, threshold):
    correct = 0
    y_pred_classes = []
    for prediction in y_pred :
      if prediction >= threshold :
        y_pred_classes.append(1)
      else :
        y_pred_classes.append(0)
    for i in range(len(y_true)):
      if y_true[i] == y_pred_classes[i]:
        correct += 1
    return correct / float(len(y_true)) * 100.0


def get_auc(y_true, y_pred):
    """
    Calculate the AUC (FPR ≤ 0.1) and average precision score.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted probabilities.
    Returns:
        tuple: AUC score and average precision score.
    """
    auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
    ap = average_precision_score(y_true, y_pred)
    return auc, ap

def get_conf_matrix(y_true, y_pred_binary):
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return tn, fp, fn, tp, precision, recall


def get_auc_per_peptide(y_true, y_pred, peptides=None):
    """
    Calculate the AUC (FPR ≤ 0.1) for each unique peptide.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): y_pred probabilities.
        peptides (array-like): Peptide identifiers for each sample.

    Returns:
        dict: A dictionary with peptides as keys and their AUC scores as values.
    """
    if peptides is None:
        return None, 0
    else:
        auc_scores = {}
        seen=set()
        for peptide in peptides:

            if peptide not in seen:
                seen.add(peptide)
            else:
                continue

            mask = (peptides == peptide)

            y_true_peptide = y_true[mask]
            y_pred_peptide = y_pred[mask]

            if len(np.unique(y_true_peptide)) > 1:  # Ensure both classes are present
                auc = roc_auc_score(y_true_peptide, y_pred_peptide, max_fpr=0.1)
                auc_scores[peptide] = auc
            else:
                auc_scores[peptide] = None  # Not enough data to calculate AUC
        
        macro_auc = np.mean([auc for auc in auc_scores.values() if auc is not None])

    
    return auc_scores, macro_auc


def plot_histogram(y_pred, bins=30, save_dir=''):
    plt.hist(y_pred, bins=bins, edgecolor='black')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)  # Fix x-axis range to [0, 1]
    plt.savefig(f'{save_dir}/hist_probs.png')  # Save the figure to a file
    plt.close()

def plot_loss_curve(fold_losses_train, fold_losses_val, output_path="loss_curve.png"):
    plt.figure()
    for i, (losses_train, losses_val) in enumerate(zip(fold_losses_train, fold_losses_val)):
        plt.plot(losses_train, label=f'Fold {i+1} Train', linestyle='dashed')
        plt.plot(losses_val, label=f'Fold {i+1} Val')
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(0, 1) 
    plt.savefig(output_path)
    plt.close()

def plot_loss_curve_logocv(losses, save_dir="."):

    # Average across folds
    all_train = np.stack([losses[pep]['train_loss'] for pep in losses])
    all_val   = np.stack([losses[pep]['val_loss']   for pep in losses])

    mean_train, std_train = all_train.mean(0), all_train.std(0)
    mean_val, std_val = all_val.mean(0), all_val.std(0)
    
    epochs = np.arange(1, len(mean_train)+1)

    for pep in losses.keys():
        plt.plot(epochs, losses[pep]['val_loss'], color='gray', alpha=0.3, linestyle='dashed', label='Folds (Val.)' if pep==list(losses.keys())[0] else "")

    plt.plot(epochs, mean_train, label='Train loss', lw=2)
    plt.fill_between(epochs, mean_train-std_train, mean_train+std_train, alpha=0.2)
    plt.plot(epochs, mean_val, label='Validation loss', lw=2)
    plt.fill_between(epochs, mean_val-std_val, mean_val+std_val, alpha=0.2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/loss_curve.png')
    plt.close()


def plot_precision_recall_curve(y_true, y_pred, save_dir):
    """
    Plots the Precision-Recall curve.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted probabilities.
        output_path (str): Path to save the plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)

    plt.figure()
    plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.savefig(f'{save_dir}/precision_recall_curve.png')
    plt.close()

def plot_roc_curve(y_true, y_scores, save_dir):
    # Calcula FPR, TPR e thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
    df.to_csv(f'{save_dir}/roc_curve.csv', index=False)

    # Plota a curva ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f'{save_dir}/roc_curve.png')
    plt.close()
