"""Ex 5.7.6: throughput of the vmapped JAX checkerboard Ising (Lab 5.10 'vrun') vs batch size B.
L=64, 200 sweeps per call, best of 5 interleaved repetitions. Also: XLA CPU thread usage note."""
import sys, os, time, math
sys.path.insert(0, "../ch05/labs")
import numpy as np, jax, jax.numpy as jnp
from ising_speed import jax_half, make_masks
L = 64; NSW = 200
m0, m1 = [jnp.asarray(m) for m in make_masks(L)]
def run_raw(s, key, b, nsweep):
    def body(_, c):
        s, key = c; key, k0, k1 = jax.random.split(key, 3)
        return (jax_half(jax_half(s, k0, b, m0), k1, b, m1), key)
    return jax.lax.fori_loop(0, nsweep, body, (s, key))
vrun = jax.jit(jax.vmap(run_raw, in_axes=(0, 0, 0, None)), static_argnums=3)
print(f"jax {jax.__version__}, devices {jax.devices()}, os.cpu_count={os.cpu_count()}")
Bs = (1, 2, 4, 8, 16, 32); args = {}
for B in Bs:
    Ts = np.linspace(1.6, 3.0, B)
    S0 = jnp.ones((B, L, L), jnp.int32); K0 = jax.random.split(jax.random.PRNGKey(2), B)
    B0 = jnp.asarray(1/Ts, jnp.float32)
    t0 = time.perf_counter(); vrun(S0, K0, B0, NSW)[0].block_until_ready(); tc = time.perf_counter()-t0
    args[B] = (S0, K0, B0, tc)
best = {B: 1e30 for B in Bs}
for rep in range(5):
    for B in Bs:
        S0, K0, B0, _ = args[B]
        t0 = time.perf_counter(); vrun(S0, K0, B0, NSW)[0].block_until_ready(); dt = time.perf_counter()-t0
        best[B] = min(best[B], dt)
for B in Bs:
    ns = best[B]/(B*NSW*L*L)*1e9
    print(f"B={B:3d}: wall {best[B]*1e3:8.1f} ms per call, {ns:7.1f} ns per spin update, "
          f"throughput {B*NSW*L*L/best[B]/1e6:6.2f} Mupdates/s, speed-up vs B=1 x{best[1]/best[B]*B:5.2f}  (first call incl. compile {args[B][3]:.2f} s)")
