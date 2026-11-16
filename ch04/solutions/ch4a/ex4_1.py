"""Verification code for Ch.4 exercises 4.1.2-4.1.6 (solutions, sol_ch4a.md)."""
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar, brentq

np.set_printoptions(precision=5, suppress=True)


def build_hamiltonian(x, V):
    """Lab 4.1 FDM Hamiltonian, H = -1/2 D2 + diag(V), Dirichlet BC."""
    N = len(x); h = x[1] - x[0]
    D2 = (np.diag(np.full(N, -2.0)) + np.diag(np.ones(N - 1), 1)
          + np.diag(np.ones(N - 1), -1)) / h**2
    return -0.5 * D2 + np.diag(V)


# ---------------- 4.1.2 unit conversion ----------------
print("=== 4.1.2 ===")
Ha_eV = 27.211386245988          # CODATA 2018
eV_J = 1.602176634e-19           # exact (SI 2019)
a0_A = 0.529177210903
print(f"E_H(1s) = -0.5 Ha = {-0.5*Ha_eV:.4f} eV = {-0.5*Ha_eV*eV_J:.4e} J")
print(f"with 27.211 eV: {-0.5*27.211:.4f} eV = {-0.5*27.211*eV_J:.4e} J")
# a0 = 4 pi eps0 hbar^2 /(m_e e^2) in SI
hbar = 1.054571817e-34; me = 9.1093837015e-31; e = 1.602176634e-19
eps0 = 8.8541878128e-12
a0_SI = 4*np.pi*eps0*hbar**2/(me*e**2)
print(f"a0 (SI formula) = {a0_SI*1e10:.6f} Angstrom; in a.u. (all constants=1): 1")
Eh_SI = me*e**4/((4*np.pi*eps0)**2*hbar**2)
print(f"Hartree (SI formula) = {Eh_SI/e:.4f} eV")

# ---------------- 4.1.3 storage estimate ----------------
print("\n=== 4.1.3 ===")
Ne, p = 50, 8
npts_log10 = 3*Ne*np.log10(p)
bytes_log10 = npts_log10 + np.log10(16)
print(f"p^(3N) = 8^150 = 10^{npts_log10:.2f} complex numbers")
print(f"bytes = 16 * 8^150 = 10^{bytes_log10:.2f} B")
print(f"ratio to 1e22 B datacenters = 10^{bytes_log10-22:.2f}")
print(f"density on same grid: 8^3 * 8 B = {8**3*8} B")

# ---------------- 4.1.4 finite square well ----------------
print("\n=== 4.1.4 ===")


def finite_well_exact(V0, a):
    """Exact bound-state energies (hbar=m=1), well |x|<a depth V0."""
    z0 = a*np.sqrt(2*V0)
    roots = []
    # even: z tan z = sqrt(z0^2-z^2); odd: -z cot z = sqrt(z0^2-z^2)
    for k in range(int(z0/(np.pi/2)) + 1):
        lo = k*np.pi/2 + 1e-12; hi = min((k+1)*np.pi/2 - 1e-12, z0 - 1e-14)
        if lo >= hi:
            continue
        if k % 2 == 0:
            f = lambda z: z*np.tan(z) - np.sqrt(z0**2 - z**2)
        else:
            f = lambda z: -z/np.tan(z) - np.sqrt(z0**2 - z**2)
        if f(lo)*f(hi) < 0:
            roots.append(brentq(f, lo, hi))
    z = np.array(roots)
    return z**2/(2*a**2), z0


def finite_well_fdm(V0, a, L=8.0, N=4000):
    """Same FDM Hamiltonian as Lab 4.1, solved with the tridiagonal symmetric solver."""
    x = np.linspace(-L, L, N); h = x[1] - x[0]
    V = np.where(np.abs(x) < a, 0.0, V0)
    E = la.eigh_tridiagonal(1.0/h**2 + V, np.full(N-1, -0.5/h**2), eigvals_only=True,
                            select="v", select_range=(-1e9, V0))
    return E[E < V0]


V0, a = 20.0, 1.0
Eb = finite_well_fdm(V0, a)
Eex, z0 = finite_well_exact(V0, a)
print(f"V0={V0}, a={a}: z0=a*sqrt(2V0)={z0:.4f}, z0/(pi/2)={z0/(np.pi/2):.3f}")
print(f"  FDM bound states ({len(Eb)}):  {Eb}")
print(f"  exact (transcendental) ({len(Eex)}): {Eex}")
print("  infinite-well ref n^2 pi^2/(8a^2):", np.array([(n*np.pi)**2/(8*a**2) for n in range(1, 6)]))
for (V0s, as_) in [(5.0, 1.0), (20.0, 1.0), (80.0, 1.0), (20.0, 0.5), (20.0, 2.0)]:
    Ef = finite_well_fdm(V0s, as_)
    Ee, z0s = finite_well_exact(V0s, as_)
    print(f"  V0={V0s:5.1f} a={as_:3.1f}: z0={z0s:6.3f}  N_bound FDM={len(Ef):2d}  exact={len(Ee):2d}"
          f"  formula ceil(2 z0/pi)={int(np.ceil(2*z0s/np.pi)):2d}")

E5 = finite_well_exact(5.0, 1.0)[0]
print("  V0=5 exact energies:", E5, f"; binding of 3rd state={5.0-E5[2]:.5f}, decay length 1/kappa={1/np.sqrt(2*(5.0-E5[2])):.1f}")
print(f"  V0=20: V0-E5={20.0-Eex[-1]:.4f}, 1/kappa={1/np.sqrt(2*(20.0-Eex[-1])):.2f}")
print("  V0=5 FDM, larger box L=40, N=40000:", finite_well_fdm(5.0, 1.0, L=40.0, N=40000))

# ---------------- 4.1.5 numerical variational optimisation ----------------
print("\n=== 4.1.5 ===")
L, N = 10.0, 1000
x = np.linspace(-L, L, N); dx = x[1] - x[0]
H = build_hamiltonian(x, 0.5*x**2)
E0_grid = la.eigvalsh(H)[0]


def E_trial(a):
    phi = np.exp(-a*x**2/2)
    phi /= np.linalg.norm(phi)          # discrete normalization  sum phi^2 = 1
    return phi @ H @ phi


for a in [0.4, 0.7, 1.0, 1.5, 2.5]:
    print(f"  a={a:4.2f}  E_grid={E_trial(a):.6f}  E_analytic={a/4+1/(4*a):.6f}")
res = minimize_scalar(E_trial, bounds=(0.2, 4.0), method="bounded", options={"xatol": 1e-10})
print(f"  minimize_scalar: a*={res.x:.6f}, E={res.fun:.8f}, nfev={res.nfev}")
print(f"  grid exact ground state eigvalsh E0={E0_grid:.8f};  E(a*)-E0_grid={res.fun-E0_grid:.2e}")
# Newton on dE/da (central differences) -- Ch.1 1.6
a_k = 0.4; hstep = 1e-4
for it in range(8):
    g = (E_trial(a_k+hstep) - E_trial(a_k-hstep))/(2*hstep)
    hss = (E_trial(a_k+hstep) - 2*E_trial(a_k) + E_trial(a_k-hstep))/hstep**2
    a_k -= g/hss
    print(f"  Newton it {it+1}: a={a_k:.8f}  E={E_trial(a_k):.8f}  |g|={abs(g):.2e}")

# ---------------- 4.1.6 broken symmetry ----------------
print("\n=== 4.1.6 ===")
L, N = 10.0, 400
x = np.linspace(-L, L, N, endpoint=False); h = x[1] - x[0]
V = 0.5*x**2
# periodic forward difference
Dp = (np.roll(np.eye(N), 1, axis=1) - np.eye(N))/h      # (Dp f)_i = (f_{i+1}-f_i)/h, periodic
H_ns = -0.5*Dp@Dp + np.diag(V)
print("  periodic D+^2: symmetric?", np.allclose(H_ns, H_ns.T))
ev = la.eigvals(H_ns)
ev = ev[np.argsort(ev.real)]
print(f"  max|Im E| = {np.max(np.abs(ev.imag)):.4f};  #eigs with |Im|>1e-8: {np.sum(np.abs(ev.imag)>1e-8)} of {N}")
print("  lowest 6 eigenvalues:", np.round(ev[:6], 4))
print("  most imaginary 3:", np.round(ev[np.argsort(-np.abs(ev.imag))[:3]], 3))
# free particle (V=0) spectrum of -1/2 D+^2 : -(1/2)(e^{ikh}-1)^2/h^2
k = 2*np.pi*np.fft.fftfreq(N, d=h)
lam_th = -0.5*(np.exp(1j*k*h) - 1)**2/h**2
lam_num = la.eigvals(-0.5*Dp@Dp)
print("  free particle: max|Im| analytic = %.3f, numeric = %.3f" % (np.max(np.abs(lam_th.imag)), np.max(np.abs(lam_num.imag))))
# small-k check: -(1/2)(ik h - k^2h^2/2 ...)^2/h^2 ~ k^2/2 - i k^3 h/2
kk = k[1]
print(f"  k={kk:.4f}: exact k^2/2={kk**2/2:.5f}, D+^2 eig={lam_th[1]:.5f}")
# symmetric periodic central difference for comparison
D2c = (np.roll(np.eye(N), 1, axis=1) + np.roll(np.eye(N), -1, axis=1) - 2*np.eye(N))/h**2
Hc = -0.5*D2c + np.diag(V)
evc = la.eigvals(Hc)
print(f"  symmetric central (periodic): max|Im E|={np.max(np.abs(evc.imag)):.2e}, lowest 3 = {np.sort(evc.real)[:3]}")
# Dirichlet D+ : upper triangular
xd = np.linspace(-L, L, N); hd = xd[1]-xd[0]
Dpd = (np.diag(np.ones(N-1), 1) - np.eye(N))/hd
Hd = -0.5*Dpd@Dpd + np.diag(0.5*xd**2)
evd = la.eigvals(Hd)
print("  Dirichlet D+^2 upper triangular?", np.allclose(np.tril(Hd, -1), 0))
print(f"  max|Im E|={np.max(np.abs(evd.imag)):.2e}; eig == diag? {np.allclose(np.sort(evd.real), np.sort(np.diag(Hd)))}")
print(f"  -1/(2h^2) = {-0.5/hd**2:.3f}; lowest eig = {np.sort(evd.real)[0]:.3f} (vs exact 0.5)")
