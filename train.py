#!/usr/bin/python3

import random
import numpy as np
#import scipy as sp
import fileinput

### Symbols for switch edges/positions and signals
s0 = 0 # Only valid edge, no switch setting
sL = 1
sR = 2
sX = 3 # No switch/edge

### HMM parameters
N  = 0     # Number of states (positions, (v, e) => |V(G)| x 3)
M  = 0     # Number of possible observations
T  = 0     # Number of observations
A  = [[0]] # Transition matrix (positions, always 1 due to fixed switches)
B  = [[0]] # Observation matrix
pi = [[0]] # Initial state probability distribution
O  = [0]   # Observation sequence (signals)
C  = [[0]] # Matrix to store the c values

### Model parameters
# Graph over switches, switch x to y may be different from y to x
G     = [[0]]   # (Switches cannot be set to 0)
p     = 0.05    # Probability of faulty signal
p_inv = 1.0 - p # Probability of correct signal
NV    = 0       # |V(G)|

"""
Parses the given text file to generate data for G and observations.
"""
def read_data():
    data = fileinput.input()

    # Read number of possible observations, |V(G)| and calculate
    # number of states
    global M, sX, NV, G, O, T, N
    M = int(next(data))
    sX = M # Invalid symbol = last (0-indexed) index + 1
    NV = int(next(data))
    N = NV * 3
    G = np.matrix(np.zeros(shape = (NV, NV)))

    # Read G values
    for i in range(NV):
        values = next(data).split()
        for j in range(NV):
           G[i, j] = int(values[j])

    # Read observation sequence length
    T = int(next(data))
    O = np.array(np.zeros(T))

    # Read observation sequence
    values = next(data).split()
    for i in range(T):
        O[i] = int(values[i])

"""
Initialise the HMM.
"""
def init_hmm():
    # Init transition matrix of size N x N
    # TODO Not row stochastic
    global A, B, pi, C
    A = np.matrix(np.ones(shape = (N, N))) # Fixes switches
    # print(A)

    # Init observation matrix of size N x M
    # TODO Not row stochastic
    B = np.matrix(np.zeros(shape = (N, M)))
    B.fill(1 / 2) # Prior for all switches
    # print(B)

    # Init initial prob dist matrix of size 0 x N
    pi = np.zeros(shape = (1, N))
    pi.fill(1 / N) # Uniform prior
    pi = np.matrix(pi)
    # print(pi)

    # Init matrix for C values of size N x T
    C = np.matrix(np.zeros(shape = (N, T)))
    # print(C)

"""
Recursively calculates the c value.
s is a position tuple (vertex, edge), t is the sequence index (1-indexed).
TODO Must set sigma first, what are the observation symbols?
"""
def c(s, t):
    print("s:", s, "t:", t)
    ### Case 1
    if t == 0:
        print(">>> Case 1")
        return 1 / NV

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
    for i in range(NV):
        if G.item(v, i) != sX and i != e1 and i != e2:
            u = i
            end_ix = i
            break
    for i in range(end_ix + 1, NV):
        if G.item(v, i) != sX and i != e1 and i != e2 and i != u:
            w = i
            break
    # Incident edges to v != e (reversed order from description)
    f = (u, v)
    g = (w, v)

    print("f:", f, "g:", g)

    # Assumes only edge e1 (v) -> e2 should be considered, since e is exit edge
    e_label = G.item(e1, e2)
    f_label = G.item(v, u) # f at v
    v_switch = G.item(v, v)
    obs = O[t - 1]
    t_prev = t - 1
    s1 = (u, f)
    s2 = (w, g)

    print("e label:", e_label, "f label:", f_label, "v switch:", v_switch)

    ### Case 2
    if e_label == s0 and obs == s0:
        print(">>> Case 2")
        return (c(s1, t_prev) + c(s2, t_prev)) * p_inv

    ### Case 3
    if e_label == s0 and obs != s0:
        print(">>> Case 3")
        return (c(s1, t_prev) + c(s2, t_prev)) * p

    # TODO Does not work with the f == 0 constraint

    ### Case 4
    # if e_label == sL and v_switch == sL and obs == sL \
    #         and f_label == s0:
    if e_label == sL and v_switch == sL and obs == sL:
        print(">>> Case 4")
        return c(s1, t_prev) * p_inv

    ### Case 5
    # if e_label == sL and v_switch == sL and obs != sL \
    #         and f_label == s0:
    if e_label == sL and v_switch == sL and obs != sL:
        print(">>> Case 5")
        return c(s1, t_prev) * p

    ### Case 6
    # if e_label == sR and v_switch == sR and obs == sR \
    #         and f_label == s0:
    if e_label == sR and v_switch == sR and obs == sR:
            # and f_label == s0:
        print(">>> Case 6")
        return c(s1, t_prev) * p_inv

    ### Case 7
    # if e_label == sR and v_switch == sR and obs != sR \
    #         and f_label == s0:
    if e_label == sR and v_switch == sR and obs != sR:
        print(">>> Case 7")
        return c(s1, t_prev) * p

    ### Case 8
    if e_label == sL and v_switch == sR:
        print(">>> Case 8")
        return 0.0

    ### Case 9
    if e_label == sR and v_switch == sL:
        print(">>> Case 9")
        return 0.0

    print("No return value for s:", s, "t:", t)
    print("g:", g, "f:", f)
    print("f label at v:", f_label)
    print("v value:", v_switch)

"""
Calculates p(s, O | G, sigma)
"""
def calc_stop_obs_prob():
    prob_sum = 0.0

    # Calculate total probability for all states (positions) by
    # finding all three edges from all vertices (assume deg(v) = 3)
    # TODO Not correct
    # for t in range(1, T+1):
    t = T
    for v in range(NV):
        e = (v, 0)
        for w in range(NV):
            if G.item(v, w) != sX and w != v:
                print("pick", w, "for", v)
                e = (v, w)
                end_ix = w
                break
        prob_sum = prob_sum + c((v, e), t)
        print(v, e)

        for w in range(end_ix + 1, NV):
            if G.item(v, w) != sX and w != v:
                print("pick", w, "for", v)
                e = (v, w)
                end_ix = w
                break
        prob_sum = prob_sum + c((v, e), t)
        print(v, e)

        for w in range(end_ix + 1, NV):
            if G.item(v, w) != sX and w != v:
                print("pick", w, "for", v)
                e = (v, w)
                break
        prob_sum = prob_sum + c((v, e), t)
        print((v, e), t, prob_sum)

    return prob_sum

"""
Calculates p(O | G, sigma)
"""
def calc_obs_prob():
    # TODO Sum out the stop position
    return 0.0

"""
Proposal distribution, a Gaussion distribution centred on x.
Should match target distribution.
"""
def q(x):
    # Centre, standard deviation (width), shape
    # return np.random.normal(x)
    return np.random.randint(1, 3)

"""
Metropolis-Hastings algorithm.
Returns an array with the generated values.
"""
def metropolis_hastings():
    iters = 100000 # MH iterations
    s     = 10     # Thinning steps
    x     = 0      # Initial state
    p     = q(x)   # Proposal sample
    print("Initial proposal sample:", p)
    samples = []

    for i in range(iters):
        # xn = x + np.random.normal() # Normal proposal distribution, recent value
        xn = np.random.randint(1, 3)
        pn = q(xn) # Sample from proposal distribution

        # Accept proposal immediately if it is better than previous
        if pn >= p:
            p = pn
            x = xn
        else:
            # Posterior is the ration between proposed and previous sample
            accept = min(1.0, pn / p) # Acceptance probability
            u = np.random.rand() # Generate uniform sample
            if u < pn / p: # Accept new samples
                p = pn
                x = xn
        if i % s == 0: # Thin
            samples.append(x)

    return np.array(samples)

if __name__ == '__main__':
    random.seed()
    read_data()
    init_hmm()

    # print(calc_stop_obs_prob())

    # Should be called on the stop position
    # print(c((0, (0, 1)), T))

    # print("Generated samples:")
    # print(metropolis_hastings())
