"""Single-histogram (Ferrenberg-Swendsen 1988) reweighting with jackknife errors."""
import numpy as np


def rw_moments(E, M, N, beta0, betas):
    """Reweight <|m|>, <m^2>, <m^4>, <E> from a run at beta0 to each beta in betas.
    E: total energies, M: total magnetizations (time series). Returns dict of arrays."""
    E = np.asarray(E, float)
    m = np.abs(np.asarray(M, float)) / N
    out = {k: np.empty(len(betas)) for k in ("am", "m2", "m4", "e")}
    for k, b in enumerate(betas):
        lw = -(b - beta0) * E
        lw -= lw.max()                       # log-sum-exp guard against overflow
        w = np.exp(lw)
        Z = w.sum()
        out["am"][k] = (w * m).sum() / Z
        out["m2"][k] = (w * m**2).sum() / Z
        out["m4"][k] = (w * m**4).sum() / Z
        out["e"][k] = (w * E).sum() / Z / N
    return out


def observables(mom, betas, N):
    U4 = 1.0 - mom["m4"] / (3.0 * mom["m2"] ** 2)
    chi = betas * N * mom["m2"]                         # chi  = beta N <m^2>
    chip = betas * N * (mom["m2"] - mom["am"] ** 2)     # chi' = beta N (<m^2> - <|m|>^2)
    return U4, chi, chip


def jackknife(E, M, N, beta0, betas, nblk=20, func=None):
    """Jackknife over nblk contiguous blocks. func(mom) -> array. Returns (full, err)."""
    n = len(E) // nblk * nblk
    E, M = np.asarray(E[:n]), np.asarray(M[:n])
    full = func(rw_moments(E, M, N, beta0, betas))
    reps = []
    idx = np.arange(n).reshape(nblk, -1)
    for k in range(nblk):
        keep = np.delete(idx, k, axis=0).ravel()
        reps.append(func(rw_moments(E[keep], M[keep], N, beta0, betas)))
    reps = np.array(reps)
    err = np.sqrt((nblk - 1) / nblk * ((reps - reps.mean(0)) ** 2).sum(0))
    return full, err, reps
