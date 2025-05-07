import pandas as pd
import numpy as np
import re
import joblib
import dgl
import torch

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler

from utils.unscaled_features import get_unscaled_features

class preprocess:
    def __init__(self, polymers, MON_SMILES_POLY, BOND_SMILES_POLY, DESCRIPTORS, SCALERS, MODEL):
        self.model = MODEL
        self.poly_IDs = polymers['ID'].map(lambda x: '_'.join(map(str, map(int, x.split('_')[:2]))))
        self.sequences = polymers['sequence']
        self.unscaled_feats = get_unscaled_features(MON_SMILES_POLY, BOND_SMILES_POLY, DESCRIPTORS)
        self.scalers = joblib.load(SCALERS)
        self.scaled_feats = {}

        # Get scaled_feats by scaling unscaled_feats with SCALERS
        for type in self.unscaled_feats.keys():
            unscaled_df = pd.DataFrame(self.unscaled_feats[type]).T.reset_index().rename(columns={'index': type})
            
            is_single_row = unscaled_df.shape[0] == 1

            data = unscaled_df.iloc[:, 1:].to_numpy()
            data = data.reshape(-1, 1) if is_single_row else data

            scaled_data = self.scalers[type].transform(data)
            scaled_data = scaled_data.T if is_single_row else scaled_data

            unscaled_df.iloc[:,1:] = scaled_data
            self.scaled_feats[type] = unscaled_df.set_index(type).apply(lambda row: row.to_numpy(), axis=1).to_dict()

    def seq_to_dgl(self, sequence):
        
        monomers = re.findall(r'[A-Z][a-z]+', sequence)

        # Initialize DGL graph
        g = dgl.graph(([], []), num_nodes=len(monomers))
        
        # Featurize nodes
        node_features = [torch.tensor(self.scaled_feats['node'][monomer], dtype=torch.float32) for monomer in monomers]
        g.ndata['h'] = torch.stack(node_features)
        
        # Edges are between sequential monomers, i.e., (0->1, 1->2, etc.)
        src_nodes = list(range(len(monomers) - 1))  # Start nodes of edges
        dst_nodes = list(range(1, len(monomers)))   # End nodes of edges
        g.add_edges(src_nodes, dst_nodes)

        # Featurize edges
        edge_features = [torch.tensor(self.scaled_feats['edge']['CC'], dtype=torch.float32)] * g.number_of_edges()
        g.edata['e'] = torch.stack(edge_features)

        if self.model == 'GCN' or self.model == 'GAT':
            g = dgl.add_self_loop(g)

        return g

    def main(self):
        self.graphs = self.sequences.apply(lambda x: self.seq_to_dgl(x))
        
        return list(zip(self.poly_IDs, self.graphs))     
