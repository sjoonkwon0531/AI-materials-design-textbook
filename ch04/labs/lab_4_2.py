import numpy as np
import scipy.linalg as la
np.set_printoptions(precision=4, suppress=True)

# Lab 4.2  A minimal 1D Kohn-Sham self-consistent (SCF) loop (recovered from v1.0 session).
def kinetic(x):
    N=len(x); h=x[1]-x[0]
    main=np.full(N,-2.0); off=np.full(N-1,1.0)
    return -0.5*(np.diag(main)+np.diag(off,1)+np.diag(off,-1))/h**2

L, N = 8.0, 400
x = np.linspace(-L, L, N); dx = x[1]-x[0]
Vext = 0.5*x**2; T = kinetic(x)
Nelec, nocc = 4, 2
a_soft = 1.0
Kmat = 1.0/np.sqrt((x[:,None]-x[None,:])**2 + a_soft**2)
cx = 0.30

def density(psi):
    n=np.zeros(N)
    for i in range(nocc): n += 2.0*(psi[:,i]/np.sqrt(dx))**2
    return n

def potentials(n, lamH):
    vH = lamH*(Kmat @ n)*dx
    vx = -2.0*cx*n
    return vH, vx

def total_energy(eps, n, lamH):
    vH,vx = potentials(n,lamH)
    EH = 0.5*np.sum(n*vH)*dx
    Ex = -cx*np.sum(n**2)*dx
    band = 2.0*np.sum(eps[:nocc])
    return band - EH - np.sum(vx*n)*dx + Ex

def scf(lamH, alpha, n_iter=200, tol=1e-9, verbose=False):
    n = np.exp(-x**2); n *= Nelec/(np.sum(n)*dx)
    Ehist=[]; dnhist=[]
    eps=psi=None
    for it in range(n_iter):
        vH,vx = potentials(n,lamH)
        eps,psi = la.eigh(T + np.diag(Vext+vH+vx))
        n_new = density(psi); n_new *= Nelec/(np.sum(n_new)*dx)
        dn = np.sqrt(np.sum((n_new-n)**2)*dx)
        Ehist.append(total_energy(eps,n_new,lamH)); dnhist.append(dn)
        if verbose and it<12:
            print(f"  it {it:2d}  E={Ehist[-1]:10.6f}  |dn|={dn:.3e}  eps0={eps[0]:.5f}")
        if dn<tol: break
        n = (1-alpha)*n + alpha*n_new
    return np.array(Ehist), np.array(dnhist), n, eps, psi

if __name__ == "__main__":
    print("--- (1) Weak coupling lamH=0.5, mixing alpha=0.3 ---")
    Eh,dnh,n_conv,eps,psi = scf(lamH=0.5, alpha=0.3, verbose=True)
    print(f"\n  converged in {len(Eh)} iters,  E = {Eh[-1]:.6f}")
    print(f"  KS eigenvalues (lowest 4): {eps[:4]}")
    print(f"  electron count = {np.sum(n_conv)*dx:.6f} (target {Nelec})")
    print("\n--- (2) Strong coupling lamH=4.0 (200 iterations, no early stop) ---")
    print(" alpha   first |dn|<1e-9   |dn| after 200   verdict")
    for al in [1.0, 0.5, 0.4, 0.3, 0.15]:
        _,dn,*_ = scf(lamH=4.0, alpha=al, n_iter=200, tol=0.0)
        hit = np.where(dn<1e-9)[0]
        first = f"{hit[0]+1:6d}" if len(hit) else " never"
        verdict = "converged" if dn[-1]<1e-9 else ("sloshing (no convergence)" if not len(hit) else "transient dip, then diverges")
        print(f" {al:4.2f}   {first:>15s}   {dn[-1]:12.2e}   {verdict}")
