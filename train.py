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
G     = np.matrix([[sL, s0, sL, sR],
                   [sR, sR, sL, s0],
                   [s0, sR, sR, sL],
                   [s0, sR, sL, s0]])
p     = 0.05 # Probability of faulty signal
p_inv = 1 - p
NV    = 4    # |V(G)|

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
    for i in range(end_ix + 1, M):
        vertex = G.item(v, i)
        if not np.isnan(vertex) and vertex != e2 and vertex != u:
            w = vertex
            break
    # Incident edges to v != e (reversed order from description)
    # f = (u, v)
    f = (v, u)
    # g = (w, v)
    g = (v, w)

    # Assumes only edge e1 (v) -> e2 should be considered, since e is exit edge
    e_label = G.item(e1, e2)
    f_label = G.item(v, u) # f at v
    v_switch = G.item(v, v)
    obs = O[t - 1]
    t_prev = t - 1

    ### Case 2
    if e_label == s0 and obs == 0:
        return (c((u, f), t_prev) + c((w, g), t_prev)) * p_inv

    ### Case 3
    if e_label == s0 and obs != 0:
        return (c((u, f), t_prev) + c((w, g), t_prev)) * p

    ### Case 4
    if e_label == sL and v_switch == sL and obs == sL \
            and f_label == s0:
        return c((u, f), t_prev) * p_inv

    ### Case 5
    if e_label == sL and v_switch == sL and obs != sL \
            and f_label == s0:
        return c((u, f), t_prev) * p

    ### Case 6
    if e_label == sR and v_switch == sR and obs == sR \
            and f_label == s0:
        return c((u, f), t_prev) * p_inv

    ### Case 7
    if e_label == sR and v_switch == sR and obs != sR \
            and f_label == s0:
        return c((u, f), t_prev) * p

    ### Case 8
    if e_label == sL and v_switch == sR:
        return 0

    ### Case 9
    if e_label == sR and v_switch == sL:
        return 0

    print("No return value for (s, t):", s, t)
    print("g:", g, "f:", f)
    print("f label at v:", f_label)
    print("v value:", v_switch)

"""
Calculates p(s, O | G, sigma)
"""
def calc_stop_obs_prob():
    prob_sum = 0.0

    # Calculate total probability for all states (positions)
    for v in range(N):
        # Find the edges, assume deg(v) == 3
        e = (v, 0)
        for i in range(M):
            w = G.item(v, i)
            if not np.isnan(w) and w != v:
                e = (v, w)
                end_ix = i
                break
        prob_sum = prob_sum + c((v, e), T)

        for i in range(end_ix + 1, M):
            w = G.item(v, i)
            if not np.isnan(w) and w != v:
                e = (v, w)
                break
        prob_sum = prob_sum + c((v, e), T)

    return prob_sum

"""
Calculates p(O | G, sigma)
"""
def calc_obs_prob():
    # TODO Sum out the stop position
    return 0.0

if __name__ == '__main__':
    random.seed()
    init_hmm()
    print(calc_stop_obs_prob())

    # Should be called on the stop position
    # print(c((0, (0, 1)), 8))
    # print(c((0, (0, 1)), T)) # Works
