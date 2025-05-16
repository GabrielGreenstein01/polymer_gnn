import pandas as pd
import numpy as np
import re

import dgl
import torch

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from utils.util_functions import seq_to_dgl

class DataSet():
    def __init__(self, db_file, features, split_dataset, LABELNAME, TASK, MODEL):
        
        self._labelname = LABELNAME
        self._model = MODEL
        self._task = TASK
        
        self._mixed = split_dataset['mixed']
        del split_dataset['mixed']

        self._df = pd.read_csv(db_file)

        self._features = features
        
        self._data = {}
        self.structure_data(split_dataset)

    def structure_data(self, split_dataset):

        self._df['dgl'] = self._df.apply(lambda row: seq_to_dgl(row['ID'], row['sequence'], self._features, self._model),axis=1)
        num_labels = len(self._df[self._labelname].unique())
        
        if self._task == 'classification':
            if num_labels == 2:
                self._ntask = 1
            if num_labels > 2:
                self._ntask = num_labels
        
        for _set in split_dataset.keys():
        
            IDs = list(split_dataset[_set].keys())
            
            dgl_graphs = self._df.set_index('ID').loc[IDs, 'dgl'].reset_index(drop=True)
            labels = self._df.set_index('ID').loc[IDs, self._labelname].reset_index(drop=True)

            masks = pd.isnull(labels)
            masks = masks.apply(lambda x: torch.tensor([1], dtype=torch.float32) if not x else torch.tensor([0], dtype=torch.float32))
            
            labels = [torch.tensor([x], dtype=torch.float32) for x in labels]
                    
            self._data[_set] = list(zip(IDs, dgl_graphs, labels, masks))

            