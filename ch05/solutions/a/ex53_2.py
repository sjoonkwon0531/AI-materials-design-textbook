"""Ex 5.3.2: harmonic-TST rates and barrier error that changes the rate by 10x."""
import numpy as np
kB = 8.617333262e-5; nu = 1e13
T = 400.0
k1 = nu * np.exp(-0.45 / (kB * T)); k2 = nu * np.exp(-0.55 / (kB * T))
print(f"400 K: k(0.45)={k1:.3e} 1/s, k(0.55)={k2:.3e} 1/s, ratio={k1/k2:.2f}, kT={kB*T:.5f} eV")
for T in (300.0, 600.0, 900.0):
    print(f"T={T:.0f} K: dE for factor 10 = kT ln10 = {kB*T*np.log(10):.4f} eV; factor for 0.1 eV error = {np.exp(0.1/(kB*T)):.1f}")
