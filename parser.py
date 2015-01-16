#!/usr/bin/python3

import numpy as np
import fileinput
from graph import Graph
from hmm import HMM

"""
Methods for reading a graph from file.
"""
class Parser:
    """
    Parses the given text file to generate data for graph G
    and HMM H observations.
    """
    def read_data(self, GR, HM):
        data = fileinput.input()

        # Read number of possible observations, |V(G)| and calculate
        # number of states
        M = int(next(data))
        NV = int(next(data))
        N = NV * 3
        G = np.matrix(np.zeros(shape = (NV, NV)))

        # Read G values
        for i in range(NV):
            values = next(data).split()
            for j in range(NV):
               G[i, j] = int(values[j])

        # Read observation sequence length
        T = int(next(data))
        O = np.array(np.zeros(T))

        # Read observation sequence
        values = next(data).split()
        for i in range(T):
            O[i] = int(values[i])

        # Set the parsed data
        GR.set_data(G, NV)
        HM.set_data(N, M)
