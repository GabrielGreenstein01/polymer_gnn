import pandas as pd
import numpy as np
from iteround import saferound
import random
import re

class Sample(object):
    """
    Class that samples polymer sequences.
    """
    def __init__(self, monomers, DP, mol_dist, sampling_method, n, batch = False):
        self._monomers = monomers
        self._DP = DP
        self._dist = mol_dist
        self._sampling_method = sampling_method
        self._n = n
        self._batch = batch

        self._cum_dist = np.cumsum(self._dist)

        self._num_of_monomers = {self._monomers[i]: self._dist[i]*self._DP for i in range(len(self._monomers))}
        self._num_of_monomers = saferound(self._num_of_monomers, 0)

        if self._batch:
            self._num_of_monomers = {key: val * self._n for key, val in self._num_of_monomers.items()}
        
        self.samples = self.generate_n_samples()

    def determine_monomer(self, x, cum_dist):
        """
        Invert CDF to determine monomer 
        """
        for i, partition in enumerate(cum_dist):
            if x < partition:
                return i

    def uniform_sample(self):
        """
        Uniformly sample from monomer distribution
        """
        polymer = []
        for i in range(self._DP):
            x = np.random.uniform(0, 1)
            polymer.append(self._monomers[self.determine_monomer(x, self._cum_dist)])

        polymer = ''.join(polymer)
        
        return polymer

    def sample_wo_replacement(self):
        """
        Sample without replacement. # of each monomer in sample is computed a priori from known polymer distribution.
        """

        dist = self._num_of_monomers
    
        all_monomers = []
        for key, value in dist.items():
            all_monomers.extend([key] * int(value))
    
        random.shuffle(all_monomers)
    
        polymer = ''.join(all_monomers)
        
        return polymer


    def generate_n_samples(self):
        # Same as sample w/o replacement, but the # of each monomer is computed across all replicates instead of a single sequence
        if self._sampling_method == 'wo_replacement' and self._batch:
            sample = self.sample_wo_replacement()
    
            sample_split = re.findall('[A-Z][^A-Z]*', sample)
            x = len(sample_split) // self._n
    
            samples = [''.join(sample_split[i:i + x]) for i in range(0, len(sample_split), x)]
    
            return samples
        else:
            samples = set()
    
            while len(samples) < self._n:
                if self._sampling_method == 'uniform':
                    samples.add(self.uniform_sample())
                elif self._sampling_method == 'wo_replacement' and not self._batch:
                    samples.add(self.sample_wo_replacement())
    
            return list(samples)