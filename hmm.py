#!/usr/bin/python3

import numpy as np
from constants import Constants

class HMM:
    ### HMM parameters
    N  = 0                               # Number of states (positions, (v, e) => |V(G)| x 3)
    M  = Constants.possible_observations # Number of possible observations
    T  = Constants.observation_count     # Number of observations
    A  = [[0]]                           # Transition matrix (positions, always 1 due to fixed switches)
    B  = [[0]]                           # Observation matrix
    pi = [[0]]                           # Initial state probability distribution
    O  = [0]                             # Observation sequence (signals)
    C  = [[0]]                           # Matrix to store the c values

    GR = 0 # Graph

    def __init__(self, N, GR):
        self.N = N
        self.GR = GR
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
            if obs == Constants.s0:
                obs = np.random.randint(1, self.M)
                observations.append(obs)
            elif obs == Constants.sL or obs == Constants.sR:
                obs = Constants.s0
                observations.append(obs)

        return observations

    """
    Walks through the graph and generates possible observations,
    while obfuscating them with a certain probability.
    """
    def generate_path_observations(self, n):
        # Initial edge type
        obs = np.random.randint(0, self.M)
        observations = [obs]
        path = [] # Only for debug

        # Initialisation of path
        u = np.random.randint(0, self.GR.NV)
        for i in range(self.GR.NV):
            if i != u and self.GR.G.item(u, i) == obs:
                path.append((u, i))
                u = i

        # First observation already set
        for i in range(n - 1):
            if obs == Constants.s0:
                # Randomise obs != s0
                obs = np.random.randint(1, self.M)
                for j in range(self.GR.NV):
                    # Make sure not going back and find the edge
                    if j != u and self.GR.G.item(u, j) == obs:
                        observations.append(obs)
                        u_old = u
                        u = j
                        path.append((u_old, u))
                        break
            elif obs == Constants.sL or obs == Constants.sR:
                # Must be s0
                obs = Constants.s0
                for j in range(self.GR.NV):
                    if j != u and self.GR.G.item(u, j) == obs:
                        observations.append(obs)
                        u_old = u
                        u = j
                        path.append((u_old, u))
                        break

        observations = np.array(observations)

        print("Unmolested observations:")
        print(observations)

        # Obfuscate observations with a probability
        for i in range(n):
            r = np.random.randint(0, 100)
            if r < Constants.probability_faulty * 100:
                observations[r] = np.random.randint(0, self.M)

        return observations

    """
    Wrapper method which sets the generated sequence of observations.
    """
    def set_obserations(self):
        self.O = np.array(np.zeros(self.T))
        self.O = self.generate_path_observations(self.T)

        print("Observations:")
        print(self.O)
