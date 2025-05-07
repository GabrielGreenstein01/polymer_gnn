import pandas as pd
import os
import json
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import dgl
import torch
import random
from torch.utils.data import DataLoader

import csv


class infer():
    def __init__(self, GPU, DATALOADER, HYPERPARAMETERS, MODEL_PATH):

        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
        self._dataloader = DATALOADER
        self._exp_config = HYPERPARAMETERS
        self._model_path = MODEL_PATH

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

    def _run_an_eval_epoch(self, model, data_loader):
        ''' Utility function for running an evaluation (validation/test) epoch
        
        Args:
        model : dgllife model, Predictor with set hyperparameters
        data_loader : DataLoader, DataLoader for train, validation, or test
        
        Returns:
        metric_dict : dict, dictionary of metric names and corresponding evaluation values
        '''
        all_preds = []
        all_IDs = []
        
        model.eval()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):
                IDs, bg = batch_data
                logits = self._predict(model, bg)
        
                all_IDs.extend(IDs)
                all_preds.append(logits.detach().cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_preds = torch.sigmoid(all_preds).numpy().ravel()

        with open('results.txt', mode='w') as file:
            file.write('ID,pred\n')
            for ID, pred in zip(all_IDs, all_preds):
                file.write(f'{ID},{pred}\n')
          
        return
        
    def _predict(self, model, bg):
        ''' Utility function for moving batched graph and node/edge feats to device
        
        Args:
        model : dgllife model, Predictor with set hyperparameters
        bg : DGLGraph, batched DGLGraph
        
        Returns:
        model(bg, node_feats, edge_feats) : model moved to device
        '''
        bg = bg.to(self._device)
        if self._exp_config['model'] in ['GCN', 'GAT']:
            node_feats = bg.ndata.pop('h').to(self._device)
            return model(bg, node_feats)
        else:
            node_feats = bg.ndata.pop('h').to(self._device)
            edge_feats = bg.edata.pop('e').to(self._device)
            return model(bg, node_feats, edge_feats)
    
    def main(self):
        ''' Performs training, validation, and testing of dataset with output of metrics to centralized files'''  
        if self._device.type == 'cpu':
            model = torch.load("{}/fullmodel.pt".format(self._model_path), map_location=torch.device('cpu'))['model']
        elif self._device.type == 'cuda':
            model = torch.load("{}/fullmodel.pt".format(self._model_path), map_location=torch.device('cuda'))['model']

        model = model.to(self._device)
        self._run_an_eval_epoch(model, self._dataloader)
  
        return
