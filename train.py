#!/usr/bin/python3

import random
import numpy as np
#import scipy as sp

### Symbols for switch edges/positions and signals
s0 = 0
sL = 1
sR = 2
sX = np.nan # No switch/edge
s0L = 0
s0R = 1
sL0 = 2
sR0 = 3

### HMM parameters
N  = 12                                  # Number of states (positions, (v, e) => |V(G)| x 3)
M  = 4                                   # Number of possible observations
# T  = 8                                   # Number of observations
T  = 3                                   # Number of observations
A  = np.matrix(np.ones(shape = (N, N)))  # Transition matrix (positions, always 1 due to fixed switches)
B  = np.matrix(np.zeros(shape = (N, M))) # Observation matrix
pi = np.matrix(np.zeros(shape = (1, N))) # Initial state probability distribution
# O  = [0, 1, 3, 2, 1, 0, 2, 3]            # Observation sequence (signals)
O  = [s0, sL, s0]                        # Observation sequence (signals)

### Model parameters
# Graph over switches, switch x to y may be different from y to x
# G  = np.matrix([[sX, s0, sL, sR],
#                 [sR, sX, sL, s0],
#                 [s0, sR, sX, sL],
#                 [s0, sR, sL, sX]])
G  = np.matrix([[sL, s0, sL, sR],
                [sR, sR, sL, s0],
                [s0, sR, sR, sL],
                [s0, sR, sL, s0]])
p  = 0.05 # Probability of faulty signal
NV = 4    # |V(G)|

"""
Initialise the HMM.
"""
def init_hmm():
    # Init matrix of size N x T and fill first column
    # c = np.zeros(shape = (N, T))
    # c[0].fill(1 / N) # Uniform prior
    # c = np.matrix(c)
    # c = c.getT()
    # print(c)

    # Init transition matrix of size N x N
    # TODO Not row stochastic
    global A
    A = np.matrix(np.ones(shape = (N, N))) # Fixes switches
    # print(A)

    # Init observation matrix of size N x M
    # TODO Not row stochastic
    global B
    B = np.matrix(np.zeros(shape = (N, M)))
    B.fill(1 / 2) # Prior for all switches
    # print(B)

    # Init initial prob dist matrix of size 0 x N
    global pi
    pi = np.zeros(shape = (1, N))
    pi.fill(1 / N) # Uniform prior
    pi = np.matrix(pi)
    # print(pi)

"""
Recursively calculates the c value.
s is a position tuple (vertex, edge), t is the sequence index (1-indexed).
TODO Must set sigma first, what are the observation symbols?
"""
def c(s, t):
    ### Case 1
    if t == 0: return 1 / NV

    # Values from position tuple
    v = s[0] # vertex
    e = s[1] # edge tuple
    e1 = e[0]
    e2 = e[1]
    # Adjacent vertices to v
    u = 0
    w = 0
    end_ix = 0
    # Search through matrix to find the adjacent vertices, assume deg(v) == 3
    for i in range(M):
        vertex = G.item(v, i)
        if not np.isnan(vertex) and vertex != e1 and vertex != e2:
            u = vertex
            end_ix = i
            break
    for i in range(end_ix, M):
        vertex = G.item(v, i)
        if not np.isnan(vertex) and vertex != e1 and vertex != e2 and not vertex == u:
            w = vertex
            break
    # Incident edges to v != e
    # Reversd order from description
    # f = (u, v)
    f = (v, u)
    # g = (w, v)
    g = (v, w)

    # Assumes only edge e1 (v) -> e2 should be considered, since e is exit edge
    ### Case 2
    if G.item(e1, e2) == s0 and O[t - 1] == 0:
        return (c((u, f), t - 1) + c((w, g), t - 1)) * (1 - p)

    ### Case 3
    if G.item(e1, e2) == s0 and O[t - 1] != 0:
        return (c((u, f), t - 1) + c((w, g), t - 1)) * p

    ### Case 4
    if G.item(e1, e2) == sL and G.item(v, v) == sL and O[t - 1] == sL \
            and G.item(v, u) == s0:
        return c((u, f), t - 1) * (1 - p)

    ### Case 5
    if G.item(e1, e2) == sL and G.item(v, v) == sL and O[t - 1] != sL \
            and G.item(v, u) == s0:
        return c((u, f), t - 1) * p

    ### Case 6
    if G.item(e1, e2) == sR and G.item(v, v) == sR and O[t - 1] == sR \
            and G.item(v, u) == s0:
        return c((u, f), t - 1) * (1 - p)

    ### Case 7
    if G.item(e1, e2) == sR and G.item(v, v) == sR and O[t - 1] != sR \
            and G.item(v, u) == s0:
        return c((u, f), t - 1) * p

    ### Case 8
    if G.item(e1, e2) == sL and G.item(v, v) == sR:
        return 0

    ### Case 9
    if G.item(e1, e2) == sR and G.item(v, v) == sL:
        return 0

    print("No return value for (s, t):", s, t)
    print("g:", g, "f:", f)
    print("f label at v:", G.item(v, u))
    print("v value:", G.item(v, v))

if __name__ == '__main__':
    random.seed()
    init_hmm()

    # Should be called on the stop position
    # print(c((0, (0, 1)), 8))
    print(c((0, (0, 1)), 3))
