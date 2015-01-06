#!/usr/bin/python3

import random
import numpy as np
import scipy as sp

# HMM parameters
N  = 10 # Number of states (positions)
M  = 10 # Number of possible observations
T  = 10 # Number of observations
A  = np.matrix(np.ones(shape = (N, N))) # Transition matrix, always 1 since switches are fixed
B  = np.matrix(np.zeros(shape = (N, M))) # Emission matrix
pi = np.matrix(np.zeros(shape = (0, N))) # Initial state probability distribution

def hmm():
    # Init matrix of size S x T and fill first column
    c = np.zeros(shape = (N, T))
    c[0].fill(1 / N)
    c = np.matrix(c)
    c = c.getT()
    print(c)

if __name__ == '__main__':
    random.seed()
    hmm()
