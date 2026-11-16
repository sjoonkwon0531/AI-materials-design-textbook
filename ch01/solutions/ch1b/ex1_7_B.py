"""Ex 1.7.B: <T> for particle in a box ground state; units hbar=m=L=1 => E1 = pi^2/2."""
import numpy as np
from scipy.integrate import simpson, quad
E1 = np.pi**2/2
psi   = lambda x: np.sqrt(2)*np.sin(np.pi*x)
dpsi  = lambda x: np.sqrt(2)*np.pi*np.cos(np.pi*x)
print(f"exact E1 = {E1:.12f}")
# (1) analytic second derivative + Simpson ; (2) integration by parts 1/2 int |psi'|^2 ;
# (3) psi'' by central finite difference on the grid (what one does with tabulated psi)
for N in [10, 20, 40, 80, 160]:
    x = np.linspace(0, 1, N+1); h = 1/N
    d2_exact = -np.pi**2*psi(x)
    T1 = simpson(-0.5*psi(x)*d2_exact, x=x)
    T2 = simpson(0.5*dpsi(x)**2, x=x)
    p = psi(x); d2 = np.zeros_like(p); d2[1:-1] = (p[2:] - 2*p[1:-1] + p[:-2])/h**2
    T3 = simpson(-0.5*p*d2, x=x)
    print(f"N={N:4d}: Simpson(psi*psi'')={T1:.10f} err={abs(T1-E1):.2e} | by-parts={T2:.10f} err={abs(T2-E1):.2e} | FD psi''={T3:.10f} err={abs(T3-E1):.2e}")
v, e = quad(lambda x: 0.5*dpsi(x)**2, 0, 1); print(f"quad: {v:.15f}, err={abs(v-E1):.1e}")
# FD-Laplacian eigenvalue (tridiagonal) lowest eigenvalue for comparison
N = 40; h = 1/N; lam = (2/h**2)*(1-np.cos(np.pi*h)) / 2
print(f"N=40 discrete lowest eigenvalue (1/2)(2/h^2)(1-cos(pi h)) = {lam:.10f}")
# physical example: electron in L = 1 nm well
hbar, me, eV = 1.054571817e-34, 9.1093837015e-31, 1.602176634e-19
print(f"electron, L=1 nm: E1 = {np.pi**2*hbar**2/(2*me*(1e-9)**2)/eV:.4f} eV")
