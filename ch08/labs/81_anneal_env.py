"""Toy defect-annealing MDP shared by Lab 81_anneal (value iteration / policy iteration / Q-learning).

State  n = number of defects in a small grain, n = 0..N.
Action a = furnace temperature T_a (K) held for one time step.
Per step, a birth-death (Markov) chain:
    P(n -> n-1) = k_ann(T) * n/N          (defects reach a sink / recombine)
    P(n -> n+1) = k_gen(T) * (N-n)/N      (thermally generated defects)
    k_ann = nu exp(-E_m / kT),  k_gen = k_ann exp(-E_f / kT)
High T: fast annihilation but higher equilibrium defect level and higher energy cost.
Reward  r = -(n/N) - c(T),   c(T) = c0 (T - 300)/800.
"""
import numpy as np

kB = 8.617333e-5
N = 20
TEMPS = np.array([500.0, 700.0, 900.0, 1100.0])
E_M, E_F = 0.40, 0.12          # eV (migration barrier, formation energy)
NU = 0.8 / np.exp(-E_M / (8.617333e-5 * 1100.0))   # attempt factor: k_ann(1100 K) = 0.8
C0 = 0.01                      # energy-cost weight
GAMMA = 0.95
nS, nA = N + 1, len(TEMPS)

def rates(T):
    ka = NU * np.exp(-E_M / (kB * T))
    kg = ka * np.exp(-E_F / (kB * T))
    return ka, kg

def build_model():
    """P[a, s, s'] and expected reward R[s, a] (reward depends on current s and a)."""
    P = np.zeros((nA, nS, nS)); R = np.zeros((nS, nA))
    for a, T in enumerate(TEMPS):
        ka, kg = rates(T)
        for n in range(nS):
            pd = ka * n / N; pu = kg * (N - n) / N
            if n > 0: P[a, n, n - 1] = pd
            if n < N: P[a, n, n + 1] = pu
            P[a, n, n] = 1.0 - (pd if n > 0 else 0) - (pu if n < N else 0)
            R[n, a] = -(n / N) - C0 * (T - 300) / 800
    assert np.allclose(P.sum(-1), 1) and (P >= 0).all()
    return P, R

def step(n, a, rng):
    """Sample one transition (the 'simulator' that model-free agents see)."""
    T = TEMPS[a]; ka, kg = rates(T)
    r = -(n / N) - C0 * (T - 300) / 800
    u = rng.random()
    pd = ka * n / N; pu = kg * (N - n) / N
    if u < pd: n2 = n - 1
    elif u < pd + pu: n2 = n + 1
    else: n2 = n
    return n2, r
