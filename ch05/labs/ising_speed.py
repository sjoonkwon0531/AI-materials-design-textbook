"""Lab 5.7A - 2D Ising Metropolis: pure Python vs NumPy checkerboard vs Numba vs JAX.
Timings are machine dependent (2-core Xeon 2.8 GHz here). Correctness is checked
against the exact infinite-lattice energy (Onsager) and magnetization (Yang).
Run: python ising_speed.py   (about 2-3 min)"""
import time, random, math, json, os
import numpy as np
from numba import njit
import jax, jax.numpy as jnp
from scipy.special import ellipk

J = 1.0
TC = 2.0 / math.log(1.0 + math.sqrt(2.0))
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- exact results
def onsager_energy(T):
    b = 1.0 / T
    k = 2.0 * math.sinh(2 * b) / math.cosh(2 * b) ** 2
    K = ellipk(k * k)                      # scipy uses parameter m = k^2
    return -(1.0 / math.tanh(2 * b)) * (1 + (2 / math.pi) * (2 * math.tanh(2 * b) ** 2 - 1) * K)

def yang_m(T):
    return (1 - math.sinh(2 / T) ** -4) ** 0.125 if T < TC else 0.0

# ---------------------------------------------------------------- 1) pure Python
def sweep_python(s, L, beta, rnd):
    for _ in range(L * L):
        i = int(rnd() * L); j = int(rnd() * L)
        nb = s[(i + 1) % L][j] + s[(i - 1) % L][j] + s[i][(j + 1) % L] + s[i][(j - 1) % L]
        dE = 2 * J * s[i][j] * nb
        if dE <= 0 or rnd() < math.exp(-beta * dE):
            s[i][j] = -s[i][j]

# ---------------------------------------------------------------- 2) NumPy checkerboard
def make_masks(L):
    ii, jj = np.indices((L, L))
    return [((ii + jj) % 2 == c) for c in (0, 1)]

def sweep_numpy(s, beta, masks, rng):
    for m in masks:                          # sites of one colour never touch each other
        nb = np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)
        dE = 2 * J * s * nb
        acc = (rng.random(s.shape) < np.exp(-beta * dE)) & m
        s[acc] *= -1

# ---------------------------------------------------------------- 3) Numba
@njit(cache=True)
def seed_numba(sd):
    np.random.seed(sd)

@njit(cache=True)
def sweeps_numba_random(s, beta, nsweep):          # Ch.1 algorithm: random site
    L = s.shape[0]
    w4 = math.exp(-4 * beta * J); w8 = math.exp(-8 * beta * J)
    for _ in range(nsweep * L * L):
        i = np.random.randint(L); j = np.random.randint(L)
        nb = s[(i + 1) % L, j] + s[(i - 1) % L, j] + s[i, (j + 1) % L] + s[i, (j - 1) % L]
        x = s[i, j] * nb                            # dE = 2 J x, x in {-4,-2,0,2,4}
        if x <= 0:
            s[i, j] = -s[i, j]
        elif x == 2:
            if np.random.random() < w4: s[i, j] = -s[i, j]
        else:
            if np.random.random() < w8: s[i, j] = -s[i, j]

@njit(cache=True)
def sweeps_numba_cb(s, beta, nsweep):               # checkerboard order, table lookup
    L = s.shape[0]
    w4 = math.exp(-4 * beta * J); w8 = math.exp(-8 * beta * J)
    for _ in range(nsweep):
        for c in range(2):
            for i in range(L):
                ip = i + 1 if i + 1 < L else 0; im = i - 1 if i > 0 else L - 1
                for j in range((i + c) % 2, L, 2):
                    jp = j + 1 if j + 1 < L else 0; jm = j - 1 if j > 0 else L - 1
                    x = s[i, j] * (s[ip, j] + s[im, j] + s[i, jp] + s[i, jm])
                    if x <= 0:
                        s[i, j] = -s[i, j]
                    elif x == 2:
                        if np.random.random() < w4: s[i, j] = -s[i, j]
                    elif np.random.random() < w8:
                        s[i, j] = -s[i, j]

@njit(cache=True)
def measure_numba(s):
    L = s.shape[0]; e = 0.0; m = 0.0
    for i in range(L):
        for j in range(L):
            e -= s[i, j] * (s[(i + 1) % L, j] + s[i, (j + 1) % L]); m += s[i, j]
    return e / (L * L), m / (L * L)

# ---------------------------------------------------------------- 4) JAX
def jax_half(s, key, beta, mask):
    nb = jnp.roll(s, 1, 0) + jnp.roll(s, -1, 0) + jnp.roll(s, 1, 1) + jnp.roll(s, -1, 1)
    dE = 2 * J * s * nb
    acc = (jax.random.uniform(key, s.shape) < jnp.exp(-beta * dE)) & mask
    return jnp.where(acc, -s, s)

def make_jax_runner(L):
    m0, m1 = [jnp.asarray(m) for m in make_masks(L)]
    def run(s, key, beta, nsweep):
        def body(_, carry):
            s, key = carry
            key, k0, k1 = jax.random.split(key, 3)
            s = jax_half(s, k0, beta, m0)
            s = jax_half(s, k1, beta, m1)
            return (s, key)
        return jax.lax.fori_loop(0, nsweep, body, (s, key))
    return jax.jit(run, static_argnums=3)

def jax_measure(s):
    L = s.shape[0]
    e = -jnp.sum(s * (jnp.roll(s, 1, 0) + jnp.roll(s, 1, 1))) / (L * L)
    return e, jnp.mean(s)

# ---------------------------------------------------------------- helpers
def blocking_err(x, nb=20):
    x = np.asarray(x); n = len(x) // nb
    b = x[: n * nb].reshape(nb, n).mean(1)
    return b.mean(), b.std(ddof=1) / math.sqrt(nb)

def timeit(f, reps):
    t0 = time.perf_counter(); f(); return (time.perf_counter() - t0) / reps

if __name__ == "__main__":
    res = {}
    L = 64; N = L * L; beta = 1 / 2.4
    # ---- set up every method once (JIT compile times recorded separately)
    random.seed(1); s_py = [[1] * L for _ in range(L)]
    rng = np.random.default_rng(1); s_np = np.ones((L, L), np.int64); masks = make_masks(L)
    seed_numba(1); s_nb = np.ones((L, L), np.int64)
    t0 = time.perf_counter(); sweeps_numba_random(s_nb, beta, 1); sweeps_numba_cb(s_nb, beta, 1)
    t_numba_jit = time.perf_counter() - t0
    run = make_jax_runner(L); key = jax.random.PRNGKey(1); s_j = jnp.ones((L, L), jnp.int32)
    t0 = time.perf_counter(); run(s_j, key, beta, 500)[0].block_until_ready()
    t_jax_jit = time.perf_counter() - t0
    m0, m1 = [jnp.asarray(m) for m in make_masks(L)]
    def run_raw(s, key, b, nsweep):
        def body(_, c):
            s, key = c; key, k0, k1 = jax.random.split(key, 3)
            return (jax_half(jax_half(s, k0, b, m0), k1, b, m1), key)
        return jax.lax.fori_loop(0, nsweep, body, (s, key))
    NT = 8; Ts = np.linspace(1.6, 3.0, NT)
    vrun = jax.jit(jax.vmap(run_raw, in_axes=(0, 0, 0, None)), static_argnums=3)
    S0 = jnp.ones((NT, L, L), jnp.int32); K0 = jax.random.split(jax.random.PRNGKey(2), NT)
    B0 = jnp.asarray(1 / Ts, jnp.float32); vrun(S0, K0, B0, 200)[0].block_until_ready()
    jobs = {
        "python":       (lambda: [sweep_python(s_py, L, beta, random.random) for _ in range(10)], 10),
        "numpy_cb":     (lambda: [sweep_numpy(s_np, beta, masks, rng) for _ in range(300)], 300),
        "numba_random": (lambda: sweeps_numba_random(s_nb, beta, 1000), 1000),
        "numba_cb":     (lambda: sweeps_numba_cb(s_nb, beta, 1000), 1000),
        "jax_cb":       (lambda: run(s_j, key, beta, 500)[0].block_until_ready(), 500),
        f"jax_vmap{NT}": (lambda: vrun(S0, K0, B0, 200)[0].block_until_ready(), 200 * NT),
    }
    # interleave 5 rounds and keep the best (the machine is shared; min is the least-perturbed)
    for k in jobs: res[k] = 1e30
    for rnd in range(5):
        for k, (f, reps) in jobs.items():
            res[k] = min(res[k], timeit(f, reps) / N * 1e9)
    print(f"=== Timing: L={L} ({N} spins), T=2.4, ns per attempted spin update (best of 5) ===")
    base = res["python"]
    for k in jobs:
        print(f"  {k:13s}: {res[k]:8.1f} ns/update   speed-up vs Python x{base / res[k]:6.0f}")
    print(f"  first call incl. JIT compile: Numba {t_numba_jit:.2f} s, JAX {t_jax_jit:.2f} s")

    # ------------------------------------------------ correctness vs exact (L=64)
    print("\n=== Correctness: L=64, 1e4 equil + 5e4 meas. sweeps (Numba cb), 20 blocks ===")
    corr = []
    for T in [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0]:
        seed_numba(7); s = np.ones((L, L), np.int64); b = 1 / T
        sweeps_numba_cb(s, b, 10000)
        E = np.empty(50000); M = np.empty(50000)
        for n in range(50000):
            sweeps_numba_cb(s, b, 1); E[n], M[n] = measure_numba(s)
        e, de = blocking_err(E); m, dm = blocking_err(np.abs(M))
        corr.append((T, e, de, m, dm, onsager_energy(T), yang_m(T)))
        print(f"  T={T:.2f}: e={e:+.4f}+/-{de:.4f} (exact {onsager_energy(T):+.4f})   "
              f"|m|={m:.4f}+/-{dm:.4f} (Yang {yang_m(T):.4f})")
    # JAX: NT temperatures in ONE vmapped call; measurements accumulated inside lax.scan
    def sample(s, key, b, nmeas):
        def step(c, _):
            s, key = run_raw(c[0], c[1], b, 1)
            e, m = jax_measure(s)
            return (s, key), (e, jnp.abs(m))
        return jax.lax.scan(step, (s, key), None, length=nmeas)
    vsample = jax.jit(jax.vmap(sample, in_axes=(0, 0, 0, None)), static_argnums=3)
    print(f"\n=== JAX vmap over {NT} temperatures: 2e3 equil + 1e4 meas. sweeps ===")
    t0 = time.perf_counter()
    S, K = vrun(S0, K0, B0, 2000)
    _, (Ej, Mj) = vsample(S, K, B0, 10000)
    Ej = np.asarray(Ej, float); Mj = np.asarray(Mj, float)
    print(f"  wall time {time.perf_counter() - t0:.1f} s for {NT} x 12000 sweeps")
    jaxres = []
    for i, T in enumerate(Ts):
        e, de = blocking_err(Ej[i]); m, dm = blocking_err(Mj[i])
        jaxres.append((float(T), float(e), float(de), float(m), float(dm)))
        print(f"  T={T:.2f}: e={e:+.4f}+/-{de:.4f} (exact {onsager_energy(T):+.4f})  "
              f"|m|={m:.3f}+/-{dm:.3f} (Yang {yang_m(T):.3f})")
    json.dump({"timing": res, "numba": corr, "jax": jaxres}, open(os.path.join(OUT, "ising_speed.json"), "w"))
