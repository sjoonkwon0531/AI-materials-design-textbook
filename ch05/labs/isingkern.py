"""Shared numba kernels for the Ising labs of section 5.2 (2D and 3D, J = 1, k_B = 1)."""
import numpy as np
from numba import njit


@njit(cache=True)
def nb_seed(s):
    np.random.seed(s)


# ----------------------------------------------------------------------------- 2D
@njit(cache=True)
def energy2d(s):
    L = s.shape[0]
    e = 0
    for i in range(L):
        for j in range(L):
            e -= s[i, j] * (s[(i + 1) % L, j] + s[i, (j + 1) % L])
    return e


@njit(cache=True)
def metro_sweep2d(s, beta, E, M):
    """L*L random-site single-spin Metropolis trials. Returns updated (E, M)."""
    L = s.shape[0]
    w4 = np.exp(-4.0 * beta)
    w8 = np.exp(-8.0 * beta)
    for _ in range(L * L):
        i = np.random.randint(L)
        j = np.random.randint(L)
        nb = s[(i + 1) % L, j] + s[(i - 1) % L, j] + s[i, (j + 1) % L] + s[i, (j - 1) % L]
        dE = 2 * s[i, j] * nb
        if dE <= 0 or (dE == 4 and np.random.random() < w4) or (dE == 8 and np.random.random() < w8):
            s[i, j] = -s[i, j]
            E += dE
            M += 2 * s[i, j]
    return E, M


@njit(cache=True)
def wolff2d(s, padd, stack):
    """One Wolff cluster flip. Returns cluster size. The cluster is flipped as it grows."""
    L = s.shape[0]
    i = np.random.randint(L)
    j = np.random.randint(L)
    s0 = s[i, j]
    s[i, j] = -s0
    stack[0, 0] = i
    stack[0, 1] = j
    top = 1
    size = 1
    while top > 0:
        top -= 1
        a = stack[top, 0]
        b = stack[top, 1]
        for k in range(4):
            if k == 0:
                x, y = (a + 1) % L, b
            elif k == 1:
                x, y = (a - 1) % L, b
            elif k == 2:
                x, y = a, (b + 1) % L
            else:
                x, y = a, (b - 1) % L
            if s[x, y] == s0 and np.random.random() < padd:
                s[x, y] = -s0
                stack[top, 0] = x
                stack[top, 1] = y
                top += 1
                size += 1
    return size


@njit(cache=True)
def run_metro2d(L, beta, n_eq, n_meas, seed):
    np.random.seed(seed)
    s = np.ones((L, L), dtype=np.int64)
    for i in range(L):
        for j in range(L):
            if np.random.random() < 0.5:
                s[i, j] = -1
    E = energy2d(s)
    M = np.sum(s)
    for _ in range(n_eq):
        E, M = metro_sweep2d(s, beta, E, M)
    Es = np.empty(n_meas)
    Ms = np.empty(n_meas)
    for t in range(n_meas):
        E, M = metro_sweep2d(s, beta, E, M)
        Es[t] = E
        Ms[t] = M
    return Es, Ms


@njit(cache=True)
def run_wolff2d_sweeps(L, beta, n_eq_cl, n_meas, seed, biased=False):
    """Wolff with one measurement per 'sweep equivalent'.
    Correct version (biased=False): the number of clusters between measurements, ncl = round(N/<C>),
    is fixed once from the equilibration phase and never depends on the production trajectory.
    biased=True reproduces a classic pitfall: 'flip clusters until >= N spins were flipped, then measure'
    -- a state-dependent stopping rule that over-samples states right after large clusters."""
    np.random.seed(seed)
    N = L * L
    s = np.ones((L, L), dtype=np.int64)
    padd = 1.0 - np.exp(-2.0 * beta)
    stack = np.empty((N, 2), dtype=np.int64)
    csum = 0.0
    for _ in range(n_eq_cl):
        csum += wolff2d(s, padd, stack)
    ncl = max(1, int(round(N / (csum / n_eq_cl))))
    Es = np.empty(n_meas)
    Ms = np.empty(n_meas)
    for t in range(n_meas):
        if biased:
            c = 0
            while c < N:
                c += wolff2d(s, padd, stack)
        else:
            for _ in range(ncl):
                wolff2d(s, padd, stack)
        Es[t] = energy2d(s)
        Ms[t] = np.sum(s)
    return Es, Ms


@njit(cache=True)
def wolff2d_dE(s, padd, stack, mark, stamp, clist):
    """Wolff cluster flip that also returns the energy change (needs a membership stamp array)."""
    L = s.shape[0]
    i = np.random.randint(L)
    j = np.random.randint(L)
    s0 = s[i, j]
    s[i, j] = -s0
    mark[i, j] = stamp
    stack[0, 0] = i
    stack[0, 1] = j
    top = 1
    size = 1
    clist[0, 0] = i
    clist[0, 1] = j
    while top > 0:
        top -= 1
        a = stack[top, 0]
        b = stack[top, 1]
        for k in range(4):
            if k == 0:
                x, y = (a + 1) % L, b
            elif k == 1:
                x, y = (a - 1) % L, b
            elif k == 2:
                x, y = a, (b + 1) % L
            else:
                x, y = a, (b - 1) % L
            if s[x, y] == s0 and np.random.random() < padd:
                s[x, y] = -s0
                mark[x, y] = stamp
                stack[top, 0] = x
                stack[top, 1] = y
                top += 1
                size += 1
                clist[size - 1, 0] = x
                clist[size - 1, 1] = y
    # energy change: only bonds between the cluster and its exterior change sign
    dE = 0
    for c in range(size):
        a = clist[c, 0]
        b = clist[c, 1]
        for k in range(4):
            if k == 0:
                x, y = (a + 1) % L, b
            elif k == 1:
                x, y = (a - 1) % L, b
            elif k == 2:
                x, y = a, (b + 1) % L
            else:
                x, y = a, (b - 1) % L
            if mark[x, y] != stamp:
                # before the flip the bond energy was -s0*s[x,y]; after it is +s0*s[x,y]
                dE += 2 * s0 * s[x, y]
    return size, dE, s0


@njit(cache=True)
def run_wolff2d(L, beta, n_eq, n_meas, seed):
    """Measure after every cluster flip. Returns E, M series and cluster sizes (E, M tracked incrementally)."""
    np.random.seed(seed)
    s = np.ones((L, L), dtype=np.int64)
    padd = 1.0 - np.exp(-2.0 * beta)
    stack = np.empty((L * L, 2), dtype=np.int64)
    mark = np.zeros((L, L), dtype=np.int64)
    clist = np.empty((L * L, 2), dtype=np.int64)
    stamp = 0
    E = energy2d(s)
    M = np.sum(s)
    for _ in range(n_eq):
        stamp += 1
        c, dE, s0 = wolff2d_dE(s, padd, stack, mark, stamp, clist)
        E += dE
        M -= 2 * s0 * c
    Es = np.empty(n_meas)
    Ms = np.empty(n_meas)
    Cs = np.empty(n_meas)
    for t in range(n_meas):
        stamp += 1
        c, dE, s0 = wolff2d_dE(s, padd, stack, mark, stamp, clist)
        E += dE
        M -= 2 * s0 * c
        Cs[t] = c
        Es[t] = E
        Ms[t] = M
    return Es, Ms, Cs


# ----------------------------------------------------------------------------- 3D
@njit(cache=True)
def energy3d(s):
    L = s.shape[0]
    e = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                e -= s[i, j, k] * (s[(i + 1) % L, j, k] + s[i, (j + 1) % L, k] + s[i, j, (k + 1) % L])
    return e


@njit(cache=True)
def wolff3d(s, padd, stack):
    L = s.shape[0]
    i = np.random.randint(L)
    j = np.random.randint(L)
    k = np.random.randint(L)
    s0 = s[i, j, k]
    s[i, j, k] = -s0
    stack[0, 0] = i; stack[0, 1] = j; stack[0, 2] = k
    top = 1
    size = 1
    while top > 0:
        top -= 1
        a = stack[top, 0]; b = stack[top, 1]; c = stack[top, 2]
        for nn in range(6):
            x, y, z = a, b, c
            if nn == 0:
                x = (a + 1) % L
            elif nn == 1:
                x = (a - 1) % L
            elif nn == 2:
                y = (b + 1) % L
            elif nn == 3:
                y = (b - 1) % L
            elif nn == 4:
                z = (c + 1) % L
            else:
                z = (c - 1) % L
            if s[x, y, z] == s0 and np.random.random() < padd:
                s[x, y, z] = -s0
                stack[top, 0] = x; stack[top, 1] = y; stack[top, 2] = z
                top += 1
                size += 1
    return size


@njit(cache=True)
def run_wolff3d_sweeps(L, beta, n_eq_cl, n_meas, seed):
    """3D Wolff, one measurement every ncl = round(N/<C>) clusters (ncl fixed after equilibration)."""
    np.random.seed(seed)
    N = L * L * L
    s = np.ones((L, L, L), dtype=np.int64)
    padd = 1.0 - np.exp(-2.0 * beta)
    stack = np.empty((N, 3), dtype=np.int64)
    csum = 0.0
    for _ in range(n_eq_cl):
        csum += wolff3d(s, padd, stack)
    ncl = max(1, int(round(N / (csum / n_eq_cl))))
    Es = np.empty(n_meas)
    Ms = np.empty(n_meas)
    for t in range(n_meas):
        for _ in range(ncl):
            wolff3d(s, padd, stack)
        Es[t] = energy3d(s)
        Ms[t] = np.sum(s)
    return Es, Ms


# ----------------------------------------------------------------------------- statistics
def autocorr(x, tmax):
    """Normalized autocorrelation rho(t), t = 0..tmax, via FFT."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[: tmax + 1]
    acf /= np.arange(n, n - tmax - 1, -1)
    return acf / acf[0]


def tau_int(x, c=6.0, tmax=None):
    """Integrated autocorrelation time with Sokal's automatic window W >= c*tau_int(W).
    Convention: tau_int = 1/2 + sum_{t>=1} rho(t), so Var(mean) = 2 tau_int sigma^2 / n."""
    n = len(x)
    if tmax is None:
        tmax = min(n // 4, 200000)
    rho = autocorr(x, tmax)
    tau = 0.5
    for W in range(1, tmax + 1):
        tau += rho[W]
        if W >= c * tau:
            break
    err = tau * np.sqrt(2.0 * (2 * W + 1) / n)   # Madras-Sokal error estimate
    return tau, err, W, rho


def blocking(x):
    """Flyvbjerg-Petersen blocking: naive standard error of the mean at each blocking level."""
    x = np.asarray(x, float)
    out = []
    while len(x) >= 16:
        n = len(x)
        out.append(np.sqrt(np.var(x, ddof=1) / n))
        m = 2 * (n // 2)
        x = 0.5 * (x[0:m:2] + x[1:m:2])
    return np.array(out)
