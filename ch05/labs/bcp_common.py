"""Shared helpers for the block-copolymer DPD case study (Sec. 5.8)."""
import math
import numpy as np
from numba import njit
from dpd_core import seed, forces, run

def build_melt(nch, N, fA, box, rng, b0=0.8):
    """nch random-walk chains A_{fN}B_{(1-f)N}; returns x, v, typ, bonds"""
    box = np.asarray(box, float); n = nch * N; NA = int(round(fA * N))
    x = np.empty((n, 3)); typ = np.empty(n, np.int64); bonds = []
    for c in range(nch):
        p = rng.random(3) * box
        for k in range(N):
            i = c * N + k
            if k > 0:
                d = rng.normal(size=3); p = p + b0 * d / np.linalg.norm(d)
                bonds.append((i - 1, i))
            x[i] = p % box; typ[i] = 0 if k < NA else 1
    v = rng.normal(size=(n, 3)); v -= v.mean(0)
    return x, v, typ, np.array(bonds, np.int64)

def qvectors(box, nmax):
    """half-space integer q-vectors 2 pi n / L with 0 < |n| <= nmax (cubic box)"""
    L = box[0]; ns = []
    r = range(-nmax, nmax + 1)
    for a in r:
        for b in r:
            for c in r:
                if (a, b, c) > (0, 0, 0) and a * a + b * b + c * c <= nmax * nmax:
                    ns.append((a, b, c))
    ns = np.array(ns, float)
    return 2 * np.pi * ns / L

@njit(cache=True)
def sq_psi(x, e, qv):
    """S_psi(q) = |sum_j e_j exp(i q.r_j)|^2 / n for each q vector (e_j = +1 A, -1 B)"""
    n = x.shape[0]; out = np.empty(qv.shape[0])
    for k in range(qv.shape[0]):
        re = 0.0; im = 0.0
        for j in range(n):
            ph = qv[k, 0] * x[j, 0] + qv[k, 1] * x[j, 1] + qv[k, 2] * x[j, 2]
            re += e[j] * math.cos(ph); im += e[j] * math.sin(ph)
        out[k] = (re * re + im * im) / n
    return out

def radial(qv, s, dq=None):
    """spherical average over shells of |q| (exact shells: group equal |q|)"""
    qm = np.round(np.linalg.norm(qv, axis=1), 6)
    uq = np.unique(qm)
    return uq, np.array([s[qm == u].mean() for u in uq])

def unwrap_chains(x, nch, N, box):
    y = x.reshape(nch, N, 3).copy()
    for k in range(1, N):
        d = y[:, k] - y[:, k - 1]; d -= box * np.round(d / box); y[:, k] = y[:, k - 1] + d
    return y

def chain_stats(x, nch, N, box):
    y = unwrap_chains(x, nch, N, box)
    Re2 = np.mean(np.sum((y[:, -1] - y[:, 0]) ** 2, 1))
    cm = y.mean(1, keepdims=True); Rg2 = np.mean(np.sum((y - cm) ** 2, 2))
    b2 = np.mean(np.sum((y[:, 1:] - y[:, :-1]) ** 2, 2))
    return Re2, Rg2, b2

def intrachain_sq(x, nch, N, NA, box, qs, nrand=24, rng=None):
    """single-chain block form factors s_AA, s_BB, s_AB (per bead, i.e. divided by N),
    orientation-averaged with nrand random directions for each |q|"""
    y = unwrap_chains(x, nch, N, box)
    rng = np.random.default_rng(0) if rng is None else rng
    dirs = rng.normal(size=(nrand, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    out = np.zeros((len(qs), 3))
    for iq, q in enumerate(qs):
        ph = np.einsum('cnd,kd->kcn', y, dirs) * q                    # (nrand, nch, N)
        ea = np.exp(1j * ph[:, :, :NA]).sum(2); eb = np.exp(1j * ph[:, :, NA:]).sum(2)
        out[iq, 0] = np.mean(np.abs(ea) ** 2) / N
        out[iq, 1] = np.mean(np.abs(eb) ** 2) / N
        out[iq, 2] = np.mean((ea * np.conj(eb)).real) / N
    return out

def rpa_spsi(saa, sbb, sab, chi):
    """incompressible RPA (Leibler) for psi = phi_A - phi_B per bead: S_psi = 4 / (Gamma - 2 chi)"""
    gam = (saa + sbb + 2 * sab) / (saa * sbb - sab ** 2)
    return 4.0 / (gam - 2 * chi)
