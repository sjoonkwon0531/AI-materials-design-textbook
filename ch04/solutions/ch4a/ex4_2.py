"""Verification code for Ch.4 exercises 4.2.4 and 4.2.5 (1D Kohn-Sham SCF of Lab 4.2)."""
import numpy as np
import scipy.linalg as la

# ---- Lab 4.2 model, copied verbatim (lab_4_2.py) ----
L, N = 8.0, 400
x = np.linspace(-L, L, N); dx = x[1] - x[0]
Vext = 0.5*x**2
T = -0.5*(np.diag(np.full(N, -2.0)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1))/dx**2
Nelec, nocc = 4, 2
Kmat = 1.0/np.sqrt((x[:, None] - x[None, :])**2 + 1.0)
cx = 0.30


def g_map(n, lamH):
    """One KS step: input density -> normalised output density (the fixed-point map g)."""
    vH = lamH*(Kmat @ n)*dx; vx = -2.0*cx*n
    # T + diag(v) is symmetric tridiagonal: same matrix as Lab 4.2's eigh, solved with the
    # tridiagonal solver for the occupied orbitals only (identical eigenpairs, ~100x faster)
    eps, psi = la.eigh_tridiagonal(np.diag(T) + Vext + vH + vx, np.diag(T, 1),
                                   select="i", select_range=(0, nocc - 1))
    n_out = sum(2.0*(psi[:, i]/np.sqrt(dx))**2 for i in range(nocc))
    return n_out*Nelec/(np.sum(n_out)*dx), eps


def n_init():
    n = np.exp(-x**2); return n*Nelec/(np.sum(n)*dx)


def norm(v):
    return np.sqrt(np.sum(v**2)*dx)


def scf_linear(lamH, alpha, n_iter=400, tol=1e-9):
    n = n_init(); hist = []
    for it in range(n_iter):
        n_out, _ = g_map(n, lamH); r = n_out - n; hist.append(norm(r))
        if hist[-1] < tol:
            return it + 1, np.array(hist), n
        n = n + alpha*r
    return None, np.array(hist), n


def scf_twostep(lamH, alpha, beta, n_iter=400, tol=1e-9):
    """n_{k+1} = n_k + alpha*(beta*r_k + (1-beta)*r_{k-1}),  r = g(n)-n (exercise formula)."""
    n = n_init(); r_prev = None; hist = []
    for it in range(n_iter):
        n_out, _ = g_map(n, lamH); r = n_out - n; hist.append(norm(r))
        if hist[-1] < tol:
            return it + 1, np.array(hist)
        if r_prev is None:
            n = n + alpha*r
        else:
            n = n + alpha*(beta*r + (1 - beta)*r_prev)
        r_prev = r
    return None, np.array(hist)


def scf_pulay(lamH, alpha, m=5, n_iter=400, tol=1e-9):
    """Pulay/DIIS (= Anderson) mixing with history m."""
    n = n_init(); N_hist = []; R_hist = []; hist = []
    for it in range(n_iter):
        n_out, _ = g_map(n, lamH); r = n_out - n; hist.append(norm(r))
        if hist[-1] < tol:
            return it + 1, np.array(hist)
        N_hist.append(n.copy()); R_hist.append(r.copy())
        N_hist = N_hist[-m:]; R_hist = R_hist[-m:]
        k = len(R_hist)
        B = np.empty((k + 1, k + 1)); B[-1, :] = -1; B[:, -1] = -1; B[-1, -1] = 0
        for i in range(k):
            for j in range(k):
                B[i, j] = np.sum(R_hist[i]*R_hist[j])*dx
        rhs = np.zeros(k + 1); rhs[-1] = -1
        c = np.linalg.lstsq(B, rhs, rcond=None)[0][:k]
        n_opt = sum(ci*ni for ci, ni in zip(c, N_hist))
        r_opt = sum(ci*ri for ci, ri in zip(c, R_hist))
        n = n_opt + alpha*r_opt
    return None, np.array(hist)


def scf_pulay_state(lamH, alpha, m):
    """Run Pulay and return the final density (for parity / identity check)."""
    n = n_init(); N_hist = []; R_hist = []
    for it in range(400):
        n_out, _ = g_map(n, lamH); r = n_out - n
        if norm(r) < 1e-11:
            return n
        N_hist = (N_hist + [n.copy()])[-m:]; R_hist = (R_hist + [r.copy()])[-m:]
        k = len(R_hist)
        B = np.empty((k + 1, k + 1)); B[-1, :] = -1; B[:, -1] = -1; B[-1, -1] = 0
        B[:k, :k] = np.array([[np.sum(a*b)*dx for b in R_hist] for a in R_hist])
        rhs = np.zeros(k + 1); rhs[-1] = -1
        c = np.linalg.lstsq(B, rhs, rcond=None)[0][:k]
        n = sum(ci*ni for ci, ni in zip(c, N_hist)) + alpha*sum(ci*ri for ci, ri in zip(c, R_hist))
    return n


if __name__ == "__main__":
    print("=== 4.2.4 (a) weak coupling lamH=0.5 ===")
    for al in [0.3, 0.5, 0.7, 1.0]:
        it, h, _ = scf_linear(0.5, al)
        print(f"  alpha={al:3.1f}: iterations to |dn|<1e-9 = {it}")

    print("\n=== 4.2.4 (b) strong coupling lamH=4.0, alpha=1 |dn| history ===")
    it, h, n_last = scf_linear(4.0, 1.0, n_iter=200)
    print("  first 10 |dn|:", np.round(h[:10], 3))
    print("  last 6  |dn|:", np.round(h[-6:], 3), " converged:", it)
    # 2-cycle check: the density alternates between two shapes
    n_a, _ = g_map(n_last, 4.0); n_b, _ = g_map(n_a, 4.0)
    print(f"  2-cycle densities parity asymmetry: ||n(x)-n(-x)||={norm(n_last-n_last[::-1]):.1e}, "
          f"{norm(n_a-n_a[::-1]):.1e};  centre density n(0) alternates {n_last[N//2]:.3f} <-> {n_a[N//2]:.3f}")
    print(f"  ||g(g(n))-n|| = {norm(n_b - n_last):.2e}  vs ||g(n)-n|| = {norm(n_a - n_last):.3f}  (period-2 orbit)")

    # Jacobian of g at the converged fixed point (finite differences)
    xr = x[::-1]
    for lamH in [0.5, 4.0]:
        itc, hc, nstar = scf_linear(lamH, 0.3, n_iter=2000, tol=1e-12)
        print(f"\n  lamH={lamH}: fixed point from alpha=0.3 in {itc} its, final |dn|={hc[-1]:.1e}, "
              f"parity asymmetry ||n(x)-n(-x)||={norm(nstar-nstar[::-1]):.1e}")
        g0, _ = g_map(nstar, lamH)
        J = np.empty((N, N)); h_fd = 1e-6
        for j in range(N):
            dn = np.zeros(N); dn[j] = h_fd/np.sqrt(dx)
            gp, _ = g_map(nstar + dn, lamH)
            J[:, j] = (gp - g0)/(h_fd/np.sqrt(dx))
        lam, vec = la.eig(J)
        order = np.argsort(lam.real); lam = lam[order]; vec = vec[:, order]
        lmin = lam.real[0]; lmax = lam.real[-1]
        v0 = vec[:, 0].real; par = np.sum(v0*v0[::-1])/np.sum(v0*v0)
        print(f"  Jacobian dg/dn: most negative eig={lmin:.3f} (parity of its eigvec <v|Pv>={par:+.2f}), "
              f"largest eig={lmax:.3f}, max|Im|={np.max(np.abs(lam.imag)):.1e}")
        for al in [1.0, 0.7, 0.5, 0.3, 0.15]:
            rho = np.max(np.abs(1 - al*(1 - lam)))
            print(f"     alpha={al:4.2f}: predicted contraction rho(M)={rho:.3f}"
                  + (f"  -> ~{np.log(1e-9/3)/np.log(rho):.0f} its to 1e-9" if rho < 1 else "  -> unstable"))
        lam_store = lam
        parity = np.array([np.real(np.vdot(vec[:, i], vec[::-1, i]))/np.real(np.vdot(vec[:, i], vec[:, i]))
                           for i in range(N)])
        par_store = parity
        ev, od = lam[parity > 0], lam[parity <= 0]
        print(f"     even-parity modes: eig range [{ev.real.min():.3f}, {ev.real.max():.3f}];  "
              f"odd-parity modes: [{od.real.min():.3f}, {od.real.max():.3f}]")
        for al in [1.0, 0.5, 0.4, 0.3]:
            print(f"     alpha={al:4.2f}: rho_even={np.max(np.abs(1-al*(1-ev))):.3f}, rho_odd={np.max(np.abs(1-al*(1-od))):.3f}")
        print(f"     stability limit alpha < 2/(1-lam_min) = {2/(1-lmin):.3f};  "
              f"optimal alpha = 2/(2-lam_min-lam_max) = {2.0/(2 - lmin - lmax):.3f}")

    print("\n  alpha=0.5, lamH=4, long run with tol=1e-14 (residual history):")
    it, h, n5 = scf_linear(4.0, 0.5, n_iter=300, tol=1e-14)
    for k in [10, 20, 30, 34, 40, 50, 60, 80, 120, 200, 299]:
        print(f"     it {k:3d}: |dn|={h[k]:.2e}")
    print(f"     minimum |dn|={h.min():.2e} at iteration {np.argmin(h)+1}")
    print(f"     final parity asymmetry ||n(x)-n(-x)||={norm(n5-n5[::-1]):.3f}")

    print("\n=== 4.2.4 measured iterations at lamH=4 ===")
    for al in [1.0, 0.7, 0.6, 0.5, 0.3, 0.15]:
        it, h, _ = scf_linear(4.0, al)
        print(f"  alpha={al:4.2f}: {it}")

    print("\n=== 4.2.5 acceleration at lamH=4 ===")
    # linearised two-step recurrence per Jacobian mode mu=lam-1:
    #   e_{k+1} = (1+a*b*mu) e_k + a*(1-b)*mu e_{k-1}   ->  roots of z^2-(1+a b mu)z-a(1-b)mu
    def rho_twostep(a, b, sel):
        mu = lam_store.real[sel] - 1
        return max(np.max(np.abs(np.roots([1, -(1 + a*b*m), -a*(1 - b)*m]))) for m in mu)
    for a_, b_ in [(0.3, 1.0), (0.3, 0.5), (0.4, 1.0), (0.4, 0.5), (0.3, 1.5), (0.5, 0.5)]:
        print(f"  predicted two-step spectral radius alpha={a_} beta={b_}: even {rho_twostep(a_, b_, par_store > 0):.3f}, "
              f"odd {rho_twostep(a_, b_, par_store <= 0):.3f}")
    print("  linear alpha=0.4:", scf_linear(4.0, 0.4)[0])
    it_, h_ = scf_twostep(4.0, 0.5, 0.5, n_iter=400, tol=1e-15)
    print(f"  two-step alpha=0.5 beta=0.5 with tol=1e-15: min |dn|={h_.min():.1e} at it {np.argmin(h_)+1}, "
          f"|dn| at it 400={h_[-1]:.2e}")
    best = None
    for al in [0.3, 0.5, 0.6]:
        it, _, _ = scf_linear(4.0, al)
        print(f"  linear  alpha={al}: {it}")
    for al in [0.2, 0.3, 0.4, 0.5]:
        for be in [0.5, 1.0, 1.5, 2.0]:
            it, _ = scf_twostep(4.0, al, be)
            print(f"  two-step alpha={al} beta={be}: {it}")
    for al in [0.3, 0.5, 1.0]:
        for m in [3, 5, 8]:
            it, h = scf_pulay(4.0, al, m)
            print(f"  Pulay/DIIS alpha={al} m={m}: {it}")
    n_lin = scf_linear(4.0, 0.3, tol=1e-11)[2]
    n_pul = scf_pulay_state(4.0, 1.0, 8)
    print(f"  Pulay(a=1,m=8) fixed point vs linear(a=0.3) fixed point: ||diff||={norm(n_pul-n_lin):.1e}, "
          f"parity asym={norm(n_pul-n_pul[::-1]):.1e}")
    print("  weak coupling lamH=0.5: linear a=0.5 ->", scf_linear(0.5, 0.5)[0],
          "; Pulay a=0.5 m=5 ->", scf_pulay(0.5, 0.5, 5)[0])
