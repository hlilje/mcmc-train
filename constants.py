"""
Class containing global constants.
"""
class Constants:
    # Symbols for switch edges/positions and signals
    s0 = 0 # Only valid edge, no switch setting
    sL = 1
    sR = 2
    sX = 3 # No switch/edge

    # Graph/HMM constants
    vertex_count          = 8
    observation_count     = 10
    possible_observations = 3

    # Limits for possible switch settings
    switch_lower  = sL
    switch_higher = sR

    # Probabilities
    probability_faulty   = 0.05
    probability_correct  = 1.0 - probability_faulty
