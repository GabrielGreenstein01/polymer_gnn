import pandas as pd
import numpy as np
import json

from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit import Chem

def get_unscaled_features(MON_SMILES, BOND_SMILES, DESCRIPTORS):

    df_monomer_smiles = pd.read_csv(MON_SMILES)
    df_bonds_smiles = pd.read_csv(BOND_SMILES)
    df = pd.concat(
        [df_monomer_smiles.assign(type='node'), df_bonds_smiles.assign(type='edge')]
    ).reset_index(drop=True)

    descriptors_to_keep = pd.read_json(DESCRIPTORS).to_dict(orient='records')[0]

    unscaled_feats = {}
    
    for type in df['type'].unique():

        df_type = df[df['type'] == type]
        full_features = df_type['SMILES'].apply(
            lambda x: Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(x), missingVal=-9999, silent=True)
        )
        features = full_features.map(lambda x: np.array([x[key] for key in descriptors_to_keep[type]]))
        feats_dict = dict(zip(df_type['Molecule'], features))
        unscaled_feats[type] = feats_dict
    
    return unscaled_feats