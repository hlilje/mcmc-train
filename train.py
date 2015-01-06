#!/usr/bin/python3

import random
import numpy as np
#import scipy as sp

# Symbols for switch positions and signals
s0 = 0
sL = 1
sR = 2
sX = np.nan # No switch/edge
s0L = 0
s0R = 1
sL0 = 2
sR0 = 3

# HMM parameters
N  = 4                                   # Number of states (positions)
M  = 4                                   # Number of possible observations
T  = 8                                   # Number of observations
A  = np.matrix(np.ones(shape = (N, N)))  # Transition matrix (positions, always 1 due to fixed switches)
B  = np.matrix(np.zeros(shape = (N, M))) # Emission matrix
pi = np.matrix(np.zeros(shape = (0, N))) # Initial state probability distribution
O  = [0, 1, 3, 2, 1, 0, 2, 3]            # Observation sequence (signals)

# Model parameters
G = np.matrix([[sX, s0, sL, sR],  # Graph of track and switches
               [sR, sX, sL, s0],
               [s0, sR, sX, sL],
               [s0, sR, sL, sX]])
p = 0.05                          # Probability of faulty signal

def init_hmm():
    # Init matrix of size N x T and fill first column
    c = np.zeros(shape = (N, T))
    c[0].fill(1 / N)
    c = np.matrix(c)
    c = c.getT()
    print(c)

if __name__ == '__main__':
    random.seed()
    init_hmm()
