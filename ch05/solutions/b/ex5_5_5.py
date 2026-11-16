"""Ex 5.5.5: decaying shear wave on a 32x32 periodic D2Q9 BGK lattice -> measured viscosity."""
import sys, numpy as np
sys.path.insert(0, "../ch05/labs")
import importlib; lbm = importlib.import_module("55_lbm")
n = 32; U0 = 1e-3; k = 2*np.pi/n
X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
solid = np.zeros((n, n), bool); F0 = np.zeros((n, n))
sinky = np.sin(k*Y)
for tau in (0.6, 0.8, 1.2):
    nu = (tau-0.5)/3
    nsteps = int(round(3.0/(nu*k*k)))           # amplitude decays by e^-3
    f = lbm.feq(np.ones((n, n)), U0*sinky, np.zeros((n, n)))
    m0 = f.sum(); px0 = (f*lbm.c[:, 0, None, None]).sum()
    t = []; U = []
    for s in range(1, nsteps+1):
        f = lbm.step_numba(f, tau, F0, F0, solid)
        rho, ux, uy = lbm.macros(f, F0, F0)
        t.append(s); U.append(2*np.mean(ux*sinky))   # projection on sin(ky)
    t = np.array(t, float); U = np.array(U)
    m = t >= 10*tau                                   # skip the initial kinetic transient
    slope = np.polyfit(t[m], np.log(U[m]), 1)[0]
    nu_m = -slope/k**2
    print(f"tau={tau}: steps={nsteps}, nu_meas={nu_m:.6f}, cs2(tau-1/2)={nu:.6f} (rel.err {nu_m/nu-1:+.2e}), "
          f"cs2*tau={tau/3:.6f} (rel.diff {nu_m/(tau/3)-1:+.3f}); U(1)/U0={U[0]/U0:.6f}; "
          f"mass drift={abs(f.sum()-m0)/m0:.1e}, x-momentum={abs((f*lbm.c[:,0,None,None]).sum()):.1e}")
# k-dependence of the residual error at tau=0.8 (finer wave on a bigger lattice): error ~ k^2
for n2 in (16, 32, 64):
    tau = 0.8; nu = (tau-0.5)/3; k2 = 2*np.pi/n2
    Y2 = np.meshgrid(np.arange(n2), np.arange(n2), indexing='ij')[1]; s2 = np.sin(k2*Y2)
    sol2 = np.zeros((n2, n2), bool); Z = np.zeros((n2, n2))
    f = lbm.feq(np.ones((n2, n2)), U0*s2, Z); ns = int(round(3.0/(nu*k2*k2))); T = []; U = []
    for s in range(1, ns+1):
        f = lbm.step_numba(f, tau, Z, Z, sol2)
        _, ux, _ = lbm.macros(f, Z, Z); T.append(s); U.append(2*np.mean(ux*s2))
    T = np.array(T, float); U = np.array(U); m = T >= 8
    nu_m = -np.polyfit(T[m], np.log(U[m]), 1)[0]/k2**2
    print(f"  tau=0.8, wavelength {n2}: rel.err {nu_m/nu-1:+.3e}")
