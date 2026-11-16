"""Shared numba Metropolis kernels (2D and 3D, periodic), same rule as the text's code."""
import numpy as np
from numba import njit
@njit(cache=True)
def seed(s): np.random.seed(s)
@njit(cache=True)
def sweep2d(spins, beta):
    L = spins.shape[0]
    for _ in range(L*L):
        i = np.random.randint(L); j = np.random.randint(L)
        nb = spins[(i+1)%L, j] + spins[(i-1)%L, j] + spins[i, (j+1)%L] + spins[i, (j-1)%L]
        dE = 2*spins[i, j]*nb
        if dE <= 0 or np.random.random() < np.exp(-beta*dE): spins[i, j] = -spins[i, j]
@njit(cache=True)
def run2d(spins, beta, n_eq, n_meas, stride):
    for _ in range(n_eq): sweep2d(spins, beta)
    out = np.empty(n_meas)
    for k in range(n_meas):
        for _ in range(stride): sweep2d(spins, beta)
        out[k] = spins.sum()/spins.size
    return out
@njit(cache=True)
def sweep3d(s, beta):
    L = s.shape[0]
    for _ in range(L*L*L):
        i = np.random.randint(L); j = np.random.randint(L); k = np.random.randint(L)
        nb = (s[(i+1)%L, j, k] + s[(i-1)%L, j, k] + s[i, (j+1)%L, k] + s[i, (j-1)%L, k]
              + s[i, j, (k+1)%L] + s[i, j, (k-1)%L])
        dE = 2*s[i, j, k]*nb
        if dE <= 0 or np.random.random() < np.exp(-beta*dE): s[i, j, k] = -s[i, j, k]
@njit(cache=True)
def run3d(s, beta, n_eq, n_meas):
    for _ in range(n_eq): sweep3d(s, beta)
    out = np.empty(n_meas)
    for k in range(n_meas):
        sweep3d(s, beta); out[k] = s.sum()/s.size
    return out
# Wolff cluster update (2D) for the finite-size-scaling exercise (beats critical slowing down)
@njit(cache=True)
def wolff2d(s, p_add):
    L = s.shape[0]; N = L*L
    stack = np.empty(N, np.int64); i0 = np.random.randint(N)
    s0 = s.flat[i0]; s.flat[i0] = -s0; stack[0] = i0; top = 1; size = 1
    while top > 0:
        top -= 1; site = stack[top]; i = site // L; j = site % L
        for (a, b) in ((i+1)%L, j), ((i-1)%L, j), (i, (j+1)%L), (i, (j-1)%L):
            n = a*L + b
            if s.flat[n] == s0 and np.random.random() < p_add:
                s.flat[n] = -s0; stack[top] = n; top += 1; size += 1
    return size
@njit(cache=True)
def run_wolff2d(s, beta, n_eq, n_meas):
    p = 1.0 - np.exp(-2.0*beta)
    for _ in range(n_eq): wolff2d(s, p)
    m = np.empty(n_meas)
    for k in range(n_meas):
        wolff2d(s, p); m[k] = s.sum()/s.size
    return m
