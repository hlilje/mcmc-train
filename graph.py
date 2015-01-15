#!/usr/bin/python3

import random
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
    NV = 10 # |V(G)|

    def __init__(self):
        self.generate_graph(self.NV)
        self.set_switch_settings()

    """
    Generates a graph of size n.
    """
    def generate_graph(self, n):
        self.G = np.matrix(np.zeros(shape = (n, n)))
        self.G.fill(self.sX) # Fill with invalid

        invalid_generation = False

        # Generate values for G
        for this in range(self.NV):
            for edge in range(3):
                if self.G.item(this, edge) == self.sX and not invalid_generation:
                    (other, other_label) = (this, self.sX)
                    valid = False
                    tries = 0

                    while not valid and not invalid_generation:
                        other = random.randint(0, self.NV - 1)
                        other_label = random.randint(0, 2)

                        fv = other != this and self.G.item(this, other) == self.sX
                        if fv:
                            valid = True
                            for i in range(3):
                                if self.G.item(other, i) == this:
                                    valid = False

                        if tries == 500:
                            invalid_generation = True
                            valid = True
                        tries = tries + 1

                    if not invalid_generation:
                        self.G[other, this] = other_label
                        self.G[this, edge] = other

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
