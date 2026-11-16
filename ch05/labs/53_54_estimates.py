"""Order-of-magnitude numbers used in Estimate Boxes of Sec. 5.3 and 5.4."""
import numpy as np
kB = 8.617333262e-5
print("== Estimate 5.3: rare-event waiting times, nu=1e13 s^-1 ==")
for T in (300.0, 600.0):
    for Ea in (0.3, 0.5, 0.75, 1.0):
        k = 1e13 * np.exp(-Ea / (kB * T))
        print(f"  T={T:.0f} K Ea={Ea:.2f} eV: k={k:.2e} s^-1, <t_wait>={1/k:.2e} s, MD steps(1 fs)={1/k/1e-15:.1e}")
print("  kT(300 K)=%.4f eV, Ea/kT for 0.5 eV = %.2f" % (kB * 300, 0.5 / (kB * 300)))
print("  sensitivity: 0.1 eV barrier error at 300 K -> rate factor %.1f" % np.exp(0.1 / (kB * 300)))
print("  Ch.2 network: eigenvalues -1001.5, -1.5 -> tau_fast=%.2e, tau_slow=%.3f" % (1 / 1001.5, 1 / 1.5))
print("== Estimate 5.4: 20 nm x 20 nm DPPC bilayer patch ==")
A = 20.0 * 20.0; apl = 0.63; nw = 30
nlip = 2 * A / apl
atoms_lip = 130; beads_lip = 12
aa = nlip * atoms_lip + nlip * nw * 3
cg = nlip * beads_lip + nlip * nw / 4
print(f"  lipids={nlip:.0f}, waters={nlip*nw:.0f}")
print(f"  all-atom particles={aa:.3g}, Martini beads={cg:.3g}, ratio={aa/cg:.1f}")
print(f"  x time step 20 fs/2 fs = 10, x effective speed-up 4 -> total ~ {aa/cg*10*4:.0f}")
print(f"  1 us Martini effective = {1e-6/(4*20e-15):.2e} steps at 20 fs; all-atom 1 us at 2 fs = {1e-6/2e-15:.1e} steps")
