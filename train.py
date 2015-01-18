#!/usr/bin/python3

import random
import collections
import numpy as np
import matplotlib.pyplot as plt
from constants import Constants
from graph import Graph
from hmm import HMM

p      = Constants.probability_faulty  # Probability of faulty signal
p_comp = Constants.probability_correct # Probability of correct signal
GR     = 0                             # Graph object
HM     = 0                             # HMM object

c_recursions = 0 # Number of c recursions
c_failures   = 0 # Number of c failures

"""
Recursively calculates the c value.
s is a position tuple (vertex, edge), t is the sequence index (1-indexed).
"""
def c(s, t):
    debug = False

    global c_recursions, c_failures
    c_recursions = c_recursions + 1

    if debug: print("s:", s, "t:", t)
    ### Case 1
    if t == 0:
        if debug: print(">>> Case 1 (base case)")
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
        # if label == Constants.s0 and i != e1 and i != e2:
        if label != Constants.sX and i != e1 and i != e2:
            u = i
            end_ix = i
            break
    for i in range(end_ix + 1, GR.NV):
        label = GR.G.item(v, i)
        if label != Constants.sX and i != e1 and i != e2 and i != u:
            w = i
            break
    # Incident edges to v != e
    # TODO Switch order?
    f = (u, v)
    g = (w, v)
    # f = (v, u)
    # g = (v, w)

    if debug: print("f:", f, "g:", g)

    # Assumes only edge e1 (v) -> e2 should be considered, since e is exit edge
    e_label = GR.G.item(e1, e2)
    f_label = GR.G.item(v, u) # f at v
    v_switch = GR.G.item(v, v)
    obs = HM.O[t - 1] # O is 0-indexed, t is not
    t_prev = t - 1
    s1 = (u, f)
    s2 = (w, g)

    if debug: print("e label:", e_label, "f label:", f_label, "v switch:", v_switch)

    ### Case 2
    if e_label == Constants.s0 and obs == Constants.s0:
        if debug: print(">>> Case 2")
        return (c(s1, t_prev) + c(s2, t_prev)) * p_comp

    ### Case 3
    if e_label == Constants.s0 and obs != Constants.s0:
        if debug: print(">>> Case 3")
        return (c(s1, t_prev) + c(s2, t_prev)) * p

    # TODO Does not always work with the f == 0 constraint

    ### Case 4
    if e_label == Constants.sL and v_switch == Constants.sL and obs == Constants.sL \
            and f_label == Constants.s0:
    # if e_label == Constants.sL and v_switch == Constants.sL and obs == Constants.sL:
        if debug: print(">>> Case 4")
        return c(s1, t_prev) * p_comp

    ### Case 5
    if e_label == Constants.sL and v_switch == Constants.sL and obs != Constants.sL \
            and f_label == Constants.s0:
    # if e_label == Constants.sL and v_switch == Constants.sL and obs != Constants.sL:
        if debug: print(">>> Case 5")
        return c(s1, t_prev) * p

    ### Case 6
    if e_label == Constants.sR and v_switch == Constants.sR and obs == Constants.sR \
            and f_label == Constants.s0:
    # if e_label == Constants.sR and v_switch == Constants.sR and obs == Constants.sR:
        if debug: print(">>> Case 6")
        return c(s1, t_prev) * p_comp

    ### Case 7
    if e_label == Constants.sR and v_switch == Constants.sR and obs != Constants.sR \
            and f_label == Constants.s0:
    # if e_label == Constants.sR and v_switch == Constants.sR and obs != Constants.sR:
        if debug: print(">>> Case 7")
        return c(s1, t_prev) * p

    ### Case 8
    if e_label == Constants.sL and v_switch == Constants.sR:
        if debug: print(">>> Case 8 (impossible path)")
        return 0.0

    ### Case 9
    if e_label == Constants.sR and v_switch == Constants.sL:
        if debug: print(">>> Case 9 (impossible path)")
        return 0.0

    if debug: print("NO RETURN VALUE FOR s:", s, "t:", t)
    if debug: print("g:", g, "f:", f)
    if debug: print("f label at v:", f_label)
    if debug: print("v value:", v_switch)

    c_failures = c_failures + 1

    return 0.0 # TODO Should not be necessary

"""
Calculates p(s | G, sigma) by summing out s.
"""
def calc_stop_obs_prob():
    debug = False

    prob_sum = 0.0
    probabilities = []

    # Calculate total probability for all states (positions) by
    # finding all three edges from all vertices (assume deg(v) = 3)
    t = HM.T
    end_ix = 0
    for v in range(GR.NV):
        e = (v, 0)
        prob = 0.0
        for w in range(GR.NV):
            if GR.G.item(v, w) != Constants.sX and w != v:
                if debug: print("pick", w, "for", v)
                e = (v, w)
                end_ix = w
                break
        prob = c((v, e), t)
        probabilities.append(prob)
        prob_sum = prob_sum + prob
        if debug: print(v, e)

        for w in range(end_ix + 1, GR.NV):
            if GR.G.item(v, w) != Constants.sX and w != v:
                if debug: print("pick", w, "for", v)
                e = (v, w)
                end_ix = w
                break
        prob = c((v, e), t)
        probabilities.append(prob)
        prob_sum = prob_sum + prob
        if debug: print(v, e)

        for w in range(end_ix + 1, GR.NV):
            if GR.G.item(v, w) != Constants.sX and w != v:
                if debug: print("pick", w, "for", v)
                e = (v, w)
                break
        prob = c((v, e), t)
        probabilities.append(prob)
        prob_sum = prob_sum + prob
        if debug: print(v, e)

    highest_prob = max(probabilities)

    if debug: print("Probabilities:", probabilities)
    # print("Most probable index:", probabilities.index(max(probabilities)), \
    #         "of", len(probabilities), "states")
    if debug: print("Highest probability:", highest_prob, "of", len(probabilities), "states")

    return (prob_sum, probabilities, highest_prob)

"""
Generates a new sigma by flipping one random switch.
"""
def get_sigma_proposal(old_sigma):
    # Flip random switch
    sigma = old_sigma.copy()
    switch = np.random.randint(0, GR.NV)
    sigma[switch] = Constants.sL if sigma[switch] == Constants.sR else Constants.sR
    return sigma

"""
Calculates the probability of the given sigma.
"""
def sigma_prob(sigma):
    GR.set_switch_settings(sigma)
    return calc_stop_obs_prob()[0] # First ix is prob sum

"""
Metropolis-Hastings algorithm.
Returns an array with the set of switches with the highest probability.
"""
def metropolis_hastings(num_samples):
    burn_in       = int(num_samples / 2)               # Number of samples to discard
    iters         = num_samples + burn_in              # MH iterations
    s             = 2                                  # Thinning steps
    sigma         = GR.generate_switch_settings(GR.NV) # Generate initial sigmas
    sigma_p       = sigma_prob(sigma)                  # Start probability
    samples       = []                                 # Sampled sigmas
    probabilities = []                                 # Saved probabilities

    print("Start sigma:")
    print(sigma)
    print("Start sigma probability:", sigma_p)

    for i in range(iters):
        if i % s == 0 and iters > burn_in: # Thin and burn-in
            samples.append(sigma)
            probabilities.append(sigma_p)

        # Flip random switch
        sigma_alt = get_sigma_proposal(sigma)

        # print("Current sigma:\t", sigma)
        # print("New sigma:\t", sigma_alt)

        # Calculate probabilites and compare
        sigma_p = sigma_prob(sigma)
        sigma_alt_p = sigma_prob(sigma_alt)
        alpha = sigma_p / sigma_alt_p # Might give 0-division
        prob = min(alpha, 1)

        rand = random.random()

        if prob <= rand: # Accept better probability
        # if prob <= 1 or prob <= rand: # Accept better probability
            sigma = sigma_alt
            sigma_p = sigma_alt_p

    # Find the switch settings which was chosen most often
    # Convert to tuples for Python comparison
    tuples = [tuple(lst) for lst in samples]
    counter = collections.Counter(tuples)
    # print(counter.values())
    # print(counter.keys())
    most_common = counter.most_common(1)[0]
    print(most_common, "is most common out of", len(samples), "samples")
    most_common = list(most_common[0]) # Unpack tuples

    return most_common, probabilities

"""
Finds the most likely stop position given the probabilities.
"""
def most_likely_stop(probabilites):
    max_prob = 0.0
    max_v = np.nan # Stop vertex
    max_e = np.nan # Stop edge

    counter = 0
    end_ix = 0
    for v in range(GR.NV):
        e = (v, 0)
        for w in range(GR.NV):
            if GR.G.item(v, w) != Constants.sX and w != v:
                e = (v, w)
                end_ix = w
                break

        prob = probabilites[counter]
        if prob > max_prob:
            max_prob = prob
            max_v = v
            max_e = e
        counter = counter + 1

        for w in range(end_ix + 1, GR.NV):
            if GR.G.item(v, w) != Constants.sX and w != v:
                e = (v, w)
                end_ix = w
                break

        prob = probabilites[counter]
        if prob > max_prob:
            max_prob = prob
            max_v = v
            max_e = e
        counter = counter + 1

        for w in range(end_ix + 1, GR.NV):
            if GR.G.item(v, w) != Constants.sX and w != v:
                e = (v, w)
                break

        prob = probabilites[counter]
        if prob > max_prob:
            max_prob = prob
            max_v = v
            max_e = e
        counter = counter + 1

    return (max_prob, max_v, max_e)

if __name__ == '__main__':
    random.seed()

    GR = Graph()
    # Initialise HMM with number of states
    HM = HMM(GR.NV * HMM.M, GR)

    path = HM.P
    print("Path:")
    print(path)

    observations =  HM.O
    print("Observations:")
    print(observations)

    settings, setting_probs = metropolis_hastings(1000)
    print("Most likely switch settings:")
    print(settings)

    GR.set_switch_settings(settings)
    prob_sum, probabilities, highest_prob = calc_stop_obs_prob()
    print("Probability for estimated correct settings:", prob_sum)

    stop_pos = most_likely_stop(probabilities)
    print("Estimated stop position (p, v, e):", stop_pos)
    e = stop_pos[2]
    print("e label:", GR.G.item(e[0], e[1]))

    print("Setting probabilities:", len(setting_probs))
    plt.plot(setting_probs)
    plt.ylabel("MCMC sigma probability convergence")
    plt.show()
