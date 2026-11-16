"""Ex 10.1.2: core-time of 1 ns MD of 10,000 Si atoms with MACE-MP-0 small (17.2 ms per atom-step)."""
c = 17.2e-3; N = 1e4; S = 1e6
t = c * N * S
print(f"one step {c*N:.0f} s; total {t:.3e} core-s = {t/3600:.4g} core-h = {t/3600/24/365.25:.3g} core-years")
w = t / 1000
print(f"on 1000 cores: {w:.3e} s = {w/3600:.3g} h = {w/86400:.3g} days")
