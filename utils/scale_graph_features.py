import pandas as pd
import json
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit import Chem
import numpy as np
import re
import dgl
import torch

import joblib

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

from utils.util_functions import get_unscaled_features

# def get_unscaled_features(SMILES, DESCRIPTORS):

#     df_smiles = pd.read_csv(SMILES)
#     descriptors_to_keep = pd.read_json(DESCRIPTORS).to_dict(orient='records')[0]

#     unscaled_feats = {}
    
#     for _type in df_smiles['type'].unique():

#         df_type = df_smiles[df_smiles['type'] == _type]
#         full_features = df_type['SMILES'].apply(
#             lambda x: Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(x), missingVal=-9999, silent=True)
#         )
#         features = full_features.map(lambda x: np.array([x[key] for key in descriptors_to_keep[_type]]))
#         feats_dict = dict(zip(df_type['molecule'], features))
#         unscaled_feats[_type] = feats_dict
    
#     return unscaled_feats

def scale_features(scale_type, string, unscaled_feats):
    string_split = re.findall('[A-Z][^A-Z]*', string)
    features = [unscaled_feats[scale_type][mon] for mon in string_split]
    scaler = MinMaxScaler(feature_range=(0, 1))     
    # scaler = QuantileTransformer(output_distribution='normal', n_quantiles=len(unscaled_feats[scale_type].keys()))
    scaled_data = scaler.fit_transform(features)
    
    unique_vals, indices = np.unique(string_split, return_index=True)
    
    return scaler, dict(zip(unique_vals, np.array(scaled_data)[indices] ))

def scale(dataset, SMILES, DESCRIPTORS):
    
    unscaled_feats = get_unscaled_features(SMILES,DESCRIPTORS)
    
    # all_monomers = ''.join(dataset['train'].values())
    # all_bonds = ''.join(
    #     (len(val) - 1) * 'Amb' if 'pep' in key else (len(val) - 1) * 'Cc'
    #     for key, val in dataset['train'].items()
    # )

    # MAKE SCALE_FEATURES UPDATE UNSCALED FEATURES IN CASE THERE ARE MONOMERS NOT IN TRAIN THAT APPEAR IN VAL/TEST

    # monomer_scaler, scaled_monomers = scale_features('monomer', all_monomers, unscaled_feats)
    # bond_scaler, scaled_bonds = scale_features('bond', all_bonds, unscaled_feats)

    # scalers = {'monomer': monomer_scaler, 'bond': bond_scaler}
    # scaled_feats = {'monomer': scaled_monomers, 'bond': scaled_bonds}
    
    # return scalers, scaled_feats

    return unscaled_feats
