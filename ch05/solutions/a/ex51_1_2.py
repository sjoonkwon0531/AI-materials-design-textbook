"""Ex 5.1.1 and 5.1.2: MD cost and harmonic-TST waiting times."""
import numpy as np
# 5.1.1
dt = 2e-15; N = 1e5; rate = 1e8
for ttot, lab in [(10e-6, "10 us"), (1e-3, "1 ms")]:
    steps = ttot / dt
    sec = steps * N / rate
    print(f"5.1.1 {lab}: steps={steps:.2e}, atom-steps={steps*N:.2e}, wall={sec:.3e} s = {sec/86400:.1f} days = {sec/3.156e7:.2f} yr")
# 5.1.2
kB = 8.617e-5; nu = 1e13; Ea = 0.6
for T in (300.0, 600.0):
    x = Ea / (kB * T); k = nu * np.exp(-x)
    print(f"5.1.2 T={T:.0f}: Ea/kT={x:.3f}, k={k:.3e} 1/s, 1/k={1/k:.3e} s, MD steps(1 fs)={1/k/1e-15:.2e}")
r = np.exp(Ea / (kB * 300) - Ea / (kB * 600))
print(f"5.1.2 speed-up 300->600 K for Ea=0.6: {r:.3e}")
# competing pathway illustration: two paths, Ea1=0.6 (nu 1e13), Ea2=0.8 (nu 1e15, entropically favoured)
for T in (300.0, 600.0):
    k1 = 1e13 * np.exp(-0.6 / (kB * T)); k2 = 1e15 * np.exp(-0.8 / (kB * T))
    print(f"  two-path example T={T:.0f}: k2/k1={k2/k1:.3e}, fraction via path 2={k2/(k1+k2):.3f}")
