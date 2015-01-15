#!/usr/bin/python3

import numpy as np

"""
Class for generating graphs.
"""
class Graph:
    ### Symbols for switch edges/positions and signals
    s0 = 0 # Only valid edge, no switch setting
    sL = 1
    sR = 2
    sX = 3 # No switch/edge

    ### Model parameters
    # Graph over switches, switch x to y may be different from y to x
    G  = [[0]]
    NV = 0 # |V(G)|

    # def __init__(self):

    """
    Sets the given data to this graph.
    """
    def set_data(self, G, NV):
        self.G = G
        self.NV = NV
        self.set_switch_settings()

    """
    Generates a graph of size n.
    """
    def generate_graph(self, n):
        self.G = np.matrix(np.zeros(shape = (n, n)))

        # Read G values
        for i in range(self.NV):
            values = next(data).split()
            for j in range(self.NV):
               self.G[i, j] = int(values[j])

    """
    Wrapper method which uses MH to sample as many switch
    settings (sigmas) as given by n.
    """
    def generate_switch_settings(self, n):
        return np.random.randint(1, 3, size = self.NV)

    """
    Populates the graph G with generate switch settings.
    """
    def set_switch_settings(self):
        sigmas = self.generate_switch_settings(self.NV)

        print("Switch settings:", sigmas)

        # Populate the diagonal
        j = 0
        for i in range(self.NV):
            self.G[i, j] = sigmas[i]
            j = j + 1
