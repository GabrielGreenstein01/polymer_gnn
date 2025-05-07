import pandas as pd
import numpy as np
from iteround import saferound
import random
import re
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors, Draw

class Polymerize(object):
    def __init__(self, sample, SMILES):
        self._sample = sample # string of monomers

        sample_split = re.findall('[A-Z][^A-Z]*', self._sample) # split sample by capital letters
        self._DP = len(sample_split)
        
        sample_smiles = [SMILES[mon] for mon in sample_split]
        self._sample_mol = [Chem.MolFromSmiles(reactant) for reactant in sample_smiles]

    def reinitialize_polymer(self, product):
        if len(product) > 1:
            return "Too many products"
        else:
            product_smiles = Chem.MolToSmiles(product[0][0])
            reinit_product = Chem.MolFromSmiles(product_smiles)
            return reinit_product
        
    def initialize_reaction(self):
        rxn1 = AllChem.ReactionFromSmarts(
            "[C:0]=[CH2:1].[CH2:2]=[C:3]>>[Kr][C:0][C:1][C:3][C:2][Xe]"
        )

        A = self._sample_mol.pop(0)
        B = self._sample_mol.pop(0)
        product = rxn1.RunReactants((A, B))
        self.polymer = self.reinitialize_polymer(product)

        return product

    def propagate_reaction(self):

        rxn2 = AllChem.ReactionFromSmarts(
            "[C:0][C:1][C:2][C:3][Xe].[C:4]=[CH2:5]>>[C:0][C:1][C:2][C:3][C:4][C:5][Xe]"
        )

        A = self.polymer
        B = self._sample_mol.pop(0)
        product = rxn2.RunReactants((A, B))
        self.polymer = self.reinitialize_polymer(product)

        return product

    def terminate_reaction(self):

        if self._DP > 2:
            # terminate & remove Xe
            rxn3 = AllChem.ReactionFromSmarts(
                "[C:0][C:1][C:2][C:3][Xe].[C:4]=[CH2:5]>>[C:0][C:1][C:2][C:3][C:4][C:5]"
            )
    
            A = self.polymer
            B = self._sample_mol.pop(0)
    
            product = rxn3.RunReactants((A, B))
            self.polymer = self.reinitialize_polymer(product)
    
            # (removes Kr)
            rxn4 = AllChem.ReactionFromSmarts(
                "[C:0][C:1][C:2][C:3][Kr]>>[C:0][C:1][C:2][C:3]" 
            )
    
            A = self.polymer
            product = rxn4.RunReactants((A,))
            self.polymer = product[0][0]
        else:
            rxn3 = AllChem.ReactionFromSmarts(
                "[C:0]=[CH2:1].[CH2:2]=[C:3]>>[C:0][C:1][C:3]=[C:2]"
            )

            A = self._sample_mol.pop(0)
            B = self._sample_mol.pop(0)
            product = rxn3.RunReactants((A,B))
            self.polymer = product[0][0]

    def run_reaction(self):

        if self._DP > 2:
            self.initialize_reaction()

        if self._DP > 3:
            for i in range(self._DP - 3):
                self.propagate_reaction()
                
        self.terminate_reaction()

    def get_smiles(self):
        return Chem.MolToSmiles(self.polymer)

    def draw_diagram(self):
        return Draw.MolToImage(self.polymer, size=(500,500))        