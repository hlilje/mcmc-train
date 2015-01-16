#!/usr/bin/python3

import random
import numpy as np
from constants import Constants

"""
Class for generating graphs.
"""
class Graph:
    ### Model parameters
    # Graph over switches, switch x to y may be different from y to x
    G  = [[0]]
    NV = Constants.vertices_count # |V(G)|
    M  = Constants.possible_observations

    def __init__(self):
        self.generate_graph(self.NV)
        self.set_switch_settings()
        print("G:")
        print(self.G)

    """
    Generates a graph of size n.
    """
    def generate_graph(self, n):
        invalid_generation = True
        total_tries = 0

        while invalid_generation:
            total_tries = total_tries + 1
            self.G = np.matrix(np.zeros(shape = (n, n)))
            self.G.fill(Constants.sX) # Fill with invalid
            failed = False

            # Generate values for every vertex in G
            for i in range(self.NV):
                # Create three edges
                for j in range(self.M):
                    if not self.full_neighbours(self.G[i, :]):
                        to = 0
                        label = Constants.sX
                        valid = False
                        tries = 0

                        while not valid:
                            tries = tries + 1
                            to = random.randint(0, self.NV - 1)
                            label = self.create_label(self.G[i, :])

                            # Check that it is a viable edge
                            if i != to and self.G.item(i, to) == Constants.sX and \
                                    self.G.item(to, i) == Constants.sX and not \
                                    self.full_neighbours(self.G[to, :]):
                                valid = True
                                self.G[i, to] = label
                                # Create another label
                                self.G[to, i] = self.create_label(self.G[to, :])

                            # Hard limit on number of tries
                            if tries == 10000:
                                failed = True
                                break

            invalid_generation = failed if True else False

        print("Graph generated in", total_tries, "tries")

    """
    Helper method to create a unique label
    """
    def create_label(self, array):
        label = random.randint(0, self.M - 1)
        while label in array:
            label = random.randint(0, self.M - 1)
        return label

    """
    Helper method which checks if the given neighbour array is full
    (>= M neighbours).
    """
    def full_neighbours(self, array):
        count = 0
        for i in range(self.NV):
            if array.item(i) != Constants.sX: count = count + 1
        return count >= self.M

    """
    Wrapper method which uses MH to sample as many switch
    settings (sigmas) as given by n.
    """
    def generate_switch_settings(self, n):
        return np.random.randint(Constants.switch_lower,
                Constants.switch_higher + 1, size = self.NV)

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
