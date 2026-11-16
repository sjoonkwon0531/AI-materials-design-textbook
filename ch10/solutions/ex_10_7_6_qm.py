"""Exercise 10.7.6 (c): one-formula-unit quantum double well for the R-point tilt of MACE-MP-0 small.
V(u) = c2 u^2 + c4 u^4 per f.u. (Lab 10.13 fit), kinetic energy m_O du/dt^2 = (1/2)(2 m_O) du/dt^2.
Finite-difference Schroedinger equation; compares levels with the well depth and k_B T."""
import numpy as np
hb2 = 4.18017e-3          # hbar^2 / (1 amu A^2) in eV
kB = 8.617333e-5
M = 2 * 15.999
for name, c2, c4 in [('MACE-MP-0 small', -0.293, 6.16), ('CHGNet 0.3.0', -0.129, 4.40)]:
    u = np.linspace(-0.6, 0.6, 2401); h = u[1] - u[0]
    V = c2 * u ** 2 + c4 * u ** 4
    T = hb2 / (2 * M * h ** 2)
    H = np.diag(V + 2 * T) - T * np.eye(len(u), k=1) - T * np.eye(len(u), k=-1)
    E = np.linalg.eigvalsh(H)[:3] * 1000
    dE = -c2 ** 2 / (4 * c4) * 1000
    w = np.sqrt(-4 * c2 / M) * 521.47; hw = w / 8065.54 * 1000
    print(f"{name}: well depth {dE:.2f} meV/f.u. = k_B x {-dE/1000/kB:.1f} K; harmonic omega at the well bottom {w:.0f} cm-1,"
          f" hbar omega/2 = {hw/2:.2f} meV; levels E0 {E[0]:+.2f}, E1 {E[1]:+.2f}, E2 {E[2]:+.2f} meV (barrier top 0)")
for T in (50, 300):
    print(f"k_B T at {T} K = {kB*T*1000:.2f} meV")
