import pandas as pd
import numpy as np
import re

import dgl
import torch

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

class DataSet():
    def __init__(self, db_file, scaled_feats, split_dataset, LABELNAME, TASK, MODEL):
        
        self._labelname = LABELNAME
        self._model = MODEL
        self._task = TASK

        self._df = pd.read_csv(db_file)

        self._data = {}

        self.structure_data(scaled_feats, split_dataset)

    def seq_to_dgl(self, ID, sequence, scaled_feats, model):

        monomers = re.findall('[A-Z][^A-Z]*', sequence)
    
        if 'pep' in ID:
            bond_type = 'Amb'
        else:
            bond_type = 'Cc'
        
        # Initialize DGL graph
        g = dgl.graph(([], []), num_nodes=len(monomers))
    
        # Featurize nodes
        node_features = [
            torch.tensor(scaled_feats["monomer"][monomer], dtype=torch.float32)
            for monomer in monomers
        ]
        g.ndata["h"] = torch.stack(node_features)
    
        # Edges are between sequential monomers, i.e., (0->1, 1->2, etc.)
        src_nodes = list(range(len(monomers) - 1))  # Start nodes of edges
        dst_nodes = list(range(1, len(monomers)))  # End nodes of edges
        g.add_edges(src_nodes, dst_nodes)
    
        # Featurize edges
        edge_features = [
            torch.tensor(scaled_feats["bond"][bond_type], dtype=torch.float32)
        ] * g.number_of_edges()
        g.edata["e"] = torch.stack(edge_features)
    
        if model == "GCN" or model == "GAT":
            g = dgl.add_self_loop(g)
    
        return g
    
    def structure_data(self, scaled_feats, split_dataset):

        self._df['dgl'] = self._df.apply(lambda row: self.seq_to_dgl(row['ID'], row['sequence'], scaled_feats, self._model),axis=1)

        #### SCALE ALL SETS AT ONCE!!!
        # self.scaler = MinMaxScaler(feature_range=(0, 1))
        # self.scaler = QuantileTransformer(output_distribution='normal', n_quantiles=len(labels.unique()))
        # self.scaler = FunctionTransformer(lambda x: x)
        # self._df['scaled_labels'] = self.scaler.fit_transform(self._df[self._labelname].to_frame()).flatten()

        num_labels = len(self._df[self._labelname].unique())
        
        if self._task == 'classification':
            if num_labels == 2:
                self._ntask = 1
            if num_labels > 2:
                self._ntask = num_labels
        
        for _set in split_dataset.keys():
        
            IDs = list(split_dataset[_set].keys())
            
            dgl_graphs = self._df.set_index('ID').loc[IDs, 'dgl'].reset_index(drop=True)
            # labels = self._df.set_index('ID').loc[IDs, 'scaled_labels'].reset_index(drop=True)
            labels = self._df.set_index('ID').loc[IDs, self._labelname].reset_index(drop=True)

            masks = pd.isnull(labels)
            masks = masks.apply(lambda x: torch.tensor([1], dtype=torch.float32) if not x else torch.tensor([0], dtype=torch.float32))
            
            labels = [torch.tensor([x], dtype=torch.float32) for x in labels]
                    
            self._data[_set] = list(zip(IDs, dgl_graphs, labels, masks))

            