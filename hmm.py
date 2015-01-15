#!/usr/bin/python3

import numpy as np

from graph import Graph

class HMM:
    ### HMM parameters
    N  = 0     # Number of states (positions, (v, e) => |V(G)| x 3)
    M  = 3     # Number of possible observations
    T  = 0     # Number of observations
    A  = [[0]] # Transition matrix (positions, always 1 due to fixed switches)
    B  = [[0]] # Observation matrix
    pi = [[0]] # Initial state probability distribution
    O  = [0]   # Observation sequence (signals)
    C  = [[0]] # Matrix to store the c values

    # def __init__(self):

    """
    Sets the given data to this HMM.
    """
    def set_data(self, N):
        self.N = N
        self.set_obserations()
        self.init_hmm()

    """
    Initialise the HMM.
    """
    def init_hmm(self):
        # Init transition matrix of size N x N
        # global A, B, pi, C
        self.A = np.matrix(np.zeros(shape = (self.N, self.N)))
        # Probability 1 for fixes switches
        self.A.fill(1 / self.N) # Make row stochastic

        # Init observation matrix of size N x M
        self.B = np.matrix(np.zeros(shape = (self.N, self.M)))
        # 1 / 2 prior for all switches
        self.B.fill((1 / 2) / (1 / 2 * self.M)) # Make row stochastic

        # Init initial prob dist matrix of size 0 x N
        self.pi = np.zeros(shape = (1, self.N))
        self.pi.fill(1 / self.N) # Uniform prior, automatically row stochastic
        self.pi = np.matrix(self.pi)

        # Init matrix for C values of size N x T
        self.C = np.matrix(np.zeros(shape = (self.N, self.T)))

    """
    Generate a plausible stream of observations.
    """
    def generate_observations(self, n):
        obs = np.random.randint(0, self.M) # Initial observation
        observations = [obs]

        for i in range(n - 1):
            if obs == Graph.s0:
                obs = np.random.randint(1, self.M)
                observations.append(obs)
            elif obs == Graph.sL or obs == Graph.sR:
                obs = Graph.s0
                observations.append(obs)

        return observations

    """
    Wrapper method which sets the generated sequence of observations.
    """
    def set_obserations(self):
        self.T = 10
        self.O = np.array(np.zeros(self.T))
        self.O = self.generate_observations(self.T)

        print("Observations:", self.O)
