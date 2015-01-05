#!/usr/bin/python3

import random
import numpy as np
import scipy as sp

# HMM parameters
S = 10 # Number of positions
T = 10 # Number of transitions
N = 10 # Number of states?
mat_transition # Transition matrix
mat_emission # Emission matrix
mat_probdist # Initial state probability distribution

def hmm():
    # Init matrix of size S x T and fill first column
    c = np.zeros(shape = (S, T))
    c[0].fill(1 / N)
    c = np.matrix(c)
    c = c.getT()
    print(c)

if __name__ == '__main__':
    random.seed()
    hmm()
