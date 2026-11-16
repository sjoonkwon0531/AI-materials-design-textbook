"""4.2.4 check with the *verbatim* Lab 4.2 SCF (full eigh): alpha=0.5 at lamH=4 only dips below tol transiently."""
import numpy as np
import scipy.linalg as la

L, N = 8.0, 400
x = np.linspace(-L, L, N); dx = x[1]-x[0]
Vext = 0.5*x**2
T = -0.5*(np.diag(np.full(N, -2.0))+np.diag(np.ones(N-1), 1)+np.diag(np.ones(N-1), -1))/dx**2
Nelec, nocc = 4, 2
Kmat = 1.0/np.sqrt((x[:, None]-x[None, :])**2 + 1.0)
cx = 0.30


def scf(lamH, alpha, n_iter=200, tol=1e-9):
    n = np.exp(-x**2); n *= Nelec/(np.sum(n)*dx); dnhist = []
    for it in range(n_iter):
        vH = lamH*(Kmat @ n)*dx; vx = -2.0*cx*n
        eps, psi = la.eigh(T + np.diag(Vext+vH+vx))
        n_new = sum(2.0*(psi[:, i]/np.sqrt(dx))**2 for i in range(nocc)); n_new *= Nelec/(np.sum(n_new)*dx)
        dn = np.sqrt(np.sum((n_new-n)**2)*dx); dnhist.append(dn)
        if dn < tol:
            break
        n = (1-alpha)*n + alpha*n_new
    return np.array(dnhist)


h = scf(4.0, 0.5)
print(f"Lab tol=1e-9: stops after {len(h)} iterations, last |dn|={h[-1]:.2e}")
h = scf(4.0, 0.5, n_iter=70, tol=1e-14)
k = int(np.argmin(h))
print(f"tol=1e-14: minimum |dn|={h[k]:.2e} at iteration {k+1}; |dn| at iteration 70 = {h[-1]:.2e}; "
      f"growth ratio per iteration (last 10) = {np.mean(h[-10:]/h[-11:-1]):.3f}")
h = scf(4.0, 0.4)
print(f"alpha=0.4: {len(h)} iterations, last |dn|={h[-1]:.2e}")
