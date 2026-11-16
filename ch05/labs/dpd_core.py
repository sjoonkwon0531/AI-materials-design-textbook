"""Minimal 3D DPD engine (Groot & Warren 1997 conventions), Numba + cell lists.
Units: r_c = m = k_BT = 1. Forces:
  F^C = a_ij (1 - r) r_hat,  F^D = -gamma w(r)^2 (r_hat . v_ij) r_hat,
  F^R = sigma w(r) theta_ij / sqrt(dt) r_hat,  w(r) = 1 - r  (r < 1),
  sigma^2 = 2 gamma k_BT (fluctuation-dissipation). Bonds: F = -C r_ij (harmonic, zero rest length).
Integrator: GW modified velocity Verlet with parameter lambda."""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def seed(sd):
    np.random.seed(sd)

HS = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[-1,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1],
               [-1,1,1],[-1,0,1],[-1,-1,1],[0,-1,1],[1,-1,1]], np.int64)

@njit(cache=True)
def forces(x, v, typ, amat, gamma, sigma, dt, box, bonds, Cb, f, ptens):
    n = x.shape[0]
    nc = np.empty(3, np.int64); cl = np.empty(3)
    for d in range(3):
        nc[d] = int(box[d]); cl[d] = box[d] / nc[d]
    ncell = nc[0] * nc[1] * nc[2]
    head = -np.ones(ncell, np.int64); nxt = np.empty(n, np.int64)
    for i in range(n):
        c0 = int(x[i, 0] / cl[0]) % nc[0]; c1 = int(x[i, 1] / cl[1]) % nc[1]; c2 = int(x[i, 2] / cl[2]) % nc[2]
        c = (c0 * nc[1] + c1) * nc[2] + c2
        nxt[i] = head[c]; head[c] = i
    for i in range(n):
        f[i, 0] = 0.0; f[i, 1] = 0.0; f[i, 2] = 0.0
    for d in range(3):
        ptens[d] = 0.0
    isq = 1.0 / math.sqrt(dt)
    for c in range(ncell):
        c2 = c % nc[2]; c1 = (c // nc[2]) % nc[1]; c0 = c // (nc[1] * nc[2])
        for o in range(14):                  # half shell: self + 13 forward neighbours
            e0 = (c0 + HS[o, 0]) % nc[0]; e1 = (c1 + HS[o, 1]) % nc[1]; e2 = (c2 + HS[o, 2]) % nc[2]
            cc = (e0 * nc[1] + e1) * nc[2] + e2
            i = head[c]
            while i >= 0:
                j = nxt[i] if o == 0 else head[cc]
                while j >= 0:
                    dx = x[i, 0] - x[j, 0]; dy = x[i, 1] - x[j, 1]; dz = x[i, 2] - x[j, 2]
                    dx -= box[0] * round(dx / box[0]); dy -= box[1] * round(dy / box[1])
                    dz -= box[2] * round(dz / box[2])
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 < 1.0:
                        r = math.sqrt(r2); ex = dx / r; ey = dy / r; ez = dz / r
                        w = 1.0 - r
                        vr = (ex * (v[i, 0] - v[j, 0]) + ey * (v[i, 1] - v[j, 1])
                              + ez * (v[i, 2] - v[j, 2]))
                        fc = amat[typ[i], typ[j]] * w
                        fm = fc - gamma * w * w * vr + sigma * w * np.random.normal() * isq
                        f[i, 0] += fm * ex; f[i, 1] += fm * ey; f[i, 2] += fm * ez
                        f[j, 0] -= fm * ex; f[j, 1] -= fm * ey; f[j, 2] -= fm * ez
                        ptens[0] += fc * ex * dx; ptens[1] += fc * ey * dy; ptens[2] += fc * ez * dz
                    j = nxt[j]
                i = nxt[i]
    for b in range(bonds.shape[0]):
        i = bonds[b, 0]; j = bonds[b, 1]
        dx = x[i, 0] - x[j, 0]; dy = x[i, 1] - x[j, 1]; dz = x[i, 2] - x[j, 2]
        dx -= box[0] * round(dx / box[0]); dy -= box[1] * round(dy / box[1]); dz -= box[2] * round(dz / box[2])
        f[i, 0] -= Cb * dx; f[i, 1] -= Cb * dy; f[i, 2] -= Cb * dz
        f[j, 0] += Cb * dx; f[j, 1] += Cb * dy; f[j, 2] += Cb * dz
        ptens[0] -= Cb * dx * dx; ptens[1] -= Cb * dy * dy; ptens[2] -= Cb * dz * dz

@njit(cache=True)
def run(x, v, f, typ, amat, gamma, sigma, dt, lam, box, bonds, Cb, nstep, every, ptens):
    """GW modified velocity Verlet. Returns per-sample [kT, Pxx, Pyy, Pzz] every `every` steps.
    (kT here is the kinetic temperature sum v^2/(3(n-1)); P includes kinetic part.)"""
    n = x.shape[0]; V = box[0] * box[1] * box[2]
    nout = nstep // every
    out = np.zeros((nout, 4))
    vt = np.empty_like(v); fo = np.empty_like(f)
    for s in range(nstep):
        for i in range(n):
            for d in range(3):
                x[i, d] += dt * v[i, d] + 0.5 * dt * dt * f[i, d]
                x[i, d] -= box[d] * math.floor(x[i, d] / box[d])
                vt[i, d] = v[i, d] + lam * dt * f[i, d]
                fo[i, d] = f[i, d]
        forces(x, vt, typ, amat, gamma, sigma, dt, box, bonds, Cb, f, ptens)
        for i in range(n):
            for d in range(3):
                v[i, d] += 0.5 * dt * (fo[i, d] + f[i, d])
        if (s + 1) % every == 0:
            k = (s + 1) // every - 1
            ke = np.zeros(3)
            for i in range(n):
                for d in range(3):
                    ke[d] += v[i, d] * v[i, d]
            out[k, 0] = (ke[0] + ke[1] + ke[2]) / (3.0 * (n - 1))
            for d in range(3):
                out[k, 1 + d] = (ke[d] + ptens[d]) / V
    return out

def init_random(n, box, rng):
    x = rng.random((n, 3)) * np.asarray(box)
    v = rng.normal(size=(n, 3)); v -= v.mean(0)
    return x, v

def make_amat(a_like, da):
    return np.array([[a_like, a_like + da], [a_like + da, a_like]], float)

@njit(cache=True)
def run_field(x, v, f, typ, amat, gamma, sigma, dt, lam, box, bonds, Cb, nstep, every, ptens, h, kx,
              mode=0, w=1.0):
    """As run(), plus an external 'chemical pattern' field U_ext = -h e_j phi(x_j), e_j = +1 (A), -1 (B).
    mode 0: phi = cos(kx x) (dense pattern, period 2 pi/kx);
    mode 1: phi = Gaussian stripes of width w at pitch P = 2 pi/kx (sparse pattern, density multiplication).
    Returns per-sample [kT, Pxx, Pyy, Pzz] (field excluded from the virial)."""
    n = x.shape[0]; V = box[0] * box[1] * box[2]
    nout = nstep // every
    out = np.zeros((nout, 4))
    vt = np.empty_like(v); fo = np.empty_like(f)
    for s in range(nstep):
        for i in range(n):
            for d in range(3):
                x[i, d] += dt * v[i, d] + 0.5 * dt * dt * f[i, d]
                x[i, d] -= box[d] * math.floor(x[i, d] / box[d])
                vt[i, d] = v[i, d] + lam * dt * f[i, d]
                fo[i, d] = f[i, d]
        forces(x, vt, typ, amat, gamma, sigma, dt, box, bonds, Cb, f, ptens)
        if h != 0.0:
            for i in range(n):
                e = 1.0 if typ[i] == 0 else -1.0
                if mode == 0:
                    f[i, 0] -= h * e * kx * math.sin(kx * x[i, 0])
                else:
                    P = 2 * math.pi / kx; d = x[i, 0] - P * round(x[i, 0] / P)
                    f[i, 0] -= h * e * d / (w * w) * math.exp(-0.5 * d * d / (w * w))
        for i in range(n):
            for d in range(3):
                v[i, d] += 0.5 * dt * (fo[i, d] + f[i, d])
        if (s + 1) % every == 0:
            k = (s + 1) // every - 1
            ke = np.zeros(3)
            for i in range(n):
                for d in range(3):
                    ke[d] += v[i, d] * v[i, d]
            out[k, 0] = (ke[0] + ke[1] + ke[2]) / (3.0 * (n - 1))
            for d in range(3):
                out[k, 1 + d] = (ke[d] + ptens[d]) / V
    return out
