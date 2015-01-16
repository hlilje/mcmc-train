#!/usr/bin/python3

import random
import numpy as np
#import scipy as sp
from constants import Constants
from graph import Graph
from hmm import HMM

p      = Constants.probability_faulty  # Probability of faulty signal
p_comp = Constants.probability_correct # Probability of correct signal
GR     = 0                             # Graph object
HM     = 0                             # HMM object

"""
Recursively calculates the c value.
s is a position tuple (vertex, edge), t is the sequence index (1-indexed).
TODO Must set sigma first, what are the observation symbols?
"""
def c(s, t):
    print("s:", s, "t:", t)
    ### Case 1
    if t == 0:
        print(">>> Case 1 (base case)")
        return 1 / GR.NV

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
    # Make sure f is the 0 edge
    for i in range(GR.NV):
        label = GR.G.item(v, i)
        if label == Constants.s0 and i != e1 and i != e2:
            u = i
            end_ix = i
            break
    for i in range(end_ix + 1, GR.NV):
        label = GR.G.item(v, i)
        if label != Constants.sX and i != e1 and i != e2 and i != u:
            w = i
            break
    # Incident edges to v != e
    f = (u, v)
    g = (w, v)

    print("f:", f, "g:", g)

    # Assumes only edge e1 (v) -> e2 should be considered, since e is exit edge
    e_label = GR.G.item(e1, e2)
    f_label = GR.G.item(v, u) # f at v
    v_switch = GR.G.item(v, v)
    obs = HM.O[t - 1] # O is 0-indexed, t is not
    t_prev = t - 1
    s1 = (u, f)
    s2 = (w, g)

    print("e label:", e_label, "f label:", f_label, "v switch:", v_switch)

    ### Case 2
    if e_label == Constants.s0 and obs == Constants.s0:
        print(">>> Case 2")
        return (c(s1, t_prev) + c(s2, t_prev)) * p_comp

    ### Case 3
    if e_label == Constants.s0 and obs != Constants.s0:
        print(">>> Case 3")
        return (c(s1, t_prev) + c(s2, t_prev)) * p

    # TODO Does not work with the f == 0 constraint

    ### Case 4
    if e_label == Constants.sL and v_switch == Constants.sL and obs == Constants.sL \
            and f_label == Constants.s0:
    # if e_label == Constants.sL and v_switch == Constants.sL and obs == Constants.sL:
        print(">>> Case 4")
        return c(s1, t_prev) * p_comp

    ### Case 5
    if e_label == Constants.sL and v_switch == Constants.sL and obs != Constants.sL \
            and f_label == Constants.s0:
    # if e_label == Constants.sL and v_switch == Constants.sL and obs != Constants.sL:
        print(">>> Case 5")
        return c(s1, t_prev) * p

    ### Case 6
    if e_label == Constants.sR and v_switch == Constants.sR and obs == Constants.sR \
            and f_label == Constants.s0:
    # if e_label == Constants.sR and v_switch == Constants.sR and obs == Constants.sR:
        print(">>> Case 6")
        return c(s1, t_prev) * p_comp

    ### Case 7
    if e_label == Constants.sR and v_switch == Constants.sR and obs != Constants.sR \
            and f_label == Constants.s0:
    # if e_label == Constants.sR and v_switch == Constants.sR and obs != Constants.sR:
        print(">>> Case 7")
        return c(s1, t_prev) * p

    ### Case 8
    if e_label == Constants.sL and v_switch == Constants.sR:
        print(">>> Case 8 (impossible path)")
        return 0.0

    ### Case 9
    if e_label == Constants.sR and v_switch == Constants.sL:
        print(">>> Case 9 (impossible path)")
        return 0.0

    print("NO RETURN VALUE FOR s:", s, "t:", t)
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
    # TODO Probably not correct
    # for t in range(1, T+1):
    t = HM.T
    for v in range(GR.NV):
        e = (v, 0)
        for w in range(GR.NV):
            if GR.G.item(v, w) != Constants.sX and w != v:
                print("pick", w, "for", v)
                e = (v, w)
                end_ix = w
                break
        prob_sum = prob_sum + c((v, e), t)
        print(v, e)

        for w in range(end_ix + 1, GR.NV):
            if GR.G.item(v, w) != Constants.sX and w != v:
                print("pick", w, "for", v)
                e = (v, w)
                end_ix = w
                break
        prob_sum = prob_sum + c((v, e), t)
        print(v, e)

        for w in range(end_ix + 1, GR.NV):
            if GR.G.item(v, w) != Constants.sX and w != v:
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
    # return np.random.randint(1, 3)
    # Number of experiments, probabilities, size
    return np.argmax(np.random.multinomial(20, [1 / 2] * 2, size = 1))

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
        xn = np.random.randint(Constants.switch_lower, Constants.switch_higher + 1)
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

    GR = Graph()
    # Initialise HMM with number of states
    HM = HMM(GR.NV * HMM.M)

    print(calc_stop_obs_prob())

    # Should be called on the stop position
    print("Stop position probability:", c((6, (6, 1)), HM.T))
    print("Stop position probability:", c((2, (2, 5)), HM.T))

    # print("Generated samples:")
    # print(metropolis_hastings())
