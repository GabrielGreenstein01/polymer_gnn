import torch

import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc, hamming_loss, roc_curve, confusion_matrix, ConfusionMatrixDisplay


class Meter_v2():
    def __init__(self, mean=None, std=None):
        '''
        Initializes a Meter_v2 object
        
        Args:
        mean : torch.float32 tensor of shape (T) or None, mean of existing training labels across tasks
        std : torch.float32 tensor of shape (T) or None, std of existing training labels across tasks
        
        '''
        self.IDs = []
        self._mask = []
        self.y_pred = []
        self.y_true = []


    def update(self, IDs, y_pred, y_true, mask=None):
        '''Updates for the result of an iteration

        Args:
        y_pred : float32 tensor, predicted labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        y_true : float32 tensor, ground truth labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        mask : None or float32 tensor, binary mask indicating the existence of ground truth labels
        '''
        self.IDs.append(IDs)
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self._mask.append(torch.ones(self.y_pred[-1].shape))
        else:
            self._mask.append(mask.detach().cpu())
    
    def _finalize(self, include_IDs = False):
        '''Utility function for preparing for evaluation.

        Returns:
        mask : float32 tensor, binary mask indicating the existence of ground truth labels
        y_pred : float32 tensor, predicted labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        y_true : float32 tensor, ground truth labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        '''
        IDs = sum(self.IDs, [])
        mask = torch.cat(self._mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        return IDs, mask, y_pred, y_true

    def compute_metrics(self, loss):

        _, mask, y_pred, y_true = self._finalize()
        
        all_targets = y_true.ravel().long()
        all_preds = y_pred.argmax(dim=1).long()
        all_probs = y_pred.softmax(dim=1)

        metrics = { 'loss': np.mean(loss),
                    'ROC-AUC': roc_auc_score(all_targets, all_probs, multi_class='ovr'),
                    'F1': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
                    'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
                    'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
                    'accuracy': accuracy_score(all_targets, all_preds),
                    'confusion_matrix': confusion_matrix(all_targets, all_preds)
                }

        n_classes = all_probs.shape[1]
        
        for i in range(n_classes):
            # Create a binary label: 1 if the true label is i, else 0
            binary_targets = (all_targets.numpy() == i).astype(int)
            
            # The probability of class i for each sample
            probs_class_i = all_probs.numpy()[:, i]
            
            try:
                roc_auc = roc_auc_score(binary_targets, probs_class_i)
            except ValueError as e:
                roc_auc = None  # In case there is only one class present in binary_targets
            
            metrics[str(i) + 'vr_roc_auc'] = roc_auc


        return metrics
        