"""Lab 55_homog -- Effective properties of Cahn-Hilliard microstructures by periodic FEM homogenization.

(1) Spectral semi-implicit Cahn-Hilliard (Ch.2 Lab 2.14 scheme) -> two-phase pixel maps.
(2) Periodic P1/Q1 FEM (scikit-fem) for the cell problems of
    - scalar conductivity:  div( k (G + grad w) ) = 0,   w periodic
    - plane-strain elasticity: div( C : (E + sym grad w) ) = 0, w periodic
(3) Compare with Voigt / Reuss / Hashin-Shtrikman (2D) bounds, laminate exact check,
    Keller-Dykhne duality, and monotone (upper-bound) convergence under mesh refinement.
Deterministic: seeds fixed. Runtime ~1-2 min on 2 cores.
"""
import numpy as np, time, json
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from skfem import MeshQuad, ElementQuad1, ElementVector, Basis, BilinearForm, LinearForm, asm
from skfem.helpers import dot, grad, ddot, sym_grad, trace, eye

# ---------------------------------------------------------------- Cahn-Hilliard
def cahn_hilliard(N=128, mean=0.0, eps=0.03, dt=0.02, nsteps=800, S=2.0, seed=0, L=2*np.pi):
    k1 = np.fft.fftfreq(N, d=L/N)*2*np.pi
    KX, KY = np.meshgrid(k1, k1, indexing='ij'); K2 = KX**2+KY**2; K4 = K2**2
    denom = 1+dt*S*K2+dt*eps**2*K4; gain = 1+dt*S*K2
    rng = np.random.default_rng(seed)
    c = mean + 0.05*(2*rng.random((N, N))-1)
    chat = np.fft.fft2(c)
    for _ in range(nsteps):
        chat = (gain*chat - dt*K2*np.fft.fft2(c**3-c))/denom
        c = np.real(np.fft.ifft2(chat))
    return c

# ---------------------------------------------------------------- periodic FEM
class PeriodicCell:
    """Unit square, n x n pixels, each pixel refined into r x r Q1 elements."""
    def __init__(self, n, r=1):
        self.n, self.r = n, r
        m = n*r
        x = np.linspace(0, 1, m+1)
        self.mesh = MeshQuad.init_tensor(x, x)
        p = self.mesh.p
        I = np.rint(p[0]*m).astype(int) % m
        J = np.rint(p[1]*m).astype(int) % m
        self.master = I*m + J                       # periodic image of every node
        nn = p.shape[1]
        self.P = sp.csr_matrix((np.ones(nn), (np.arange(nn), self.master)), shape=(nn, m*m))
        cent = p[:, self.mesh.t].mean(axis=1)
        self.pix = (np.floor(cent[0]*n).astype(int).clip(0, n-1),
                    np.floor(cent[1]*n).astype(int).clip(0, n-1))
        self.basis = Basis(self.mesh, ElementQuad1())
        self.vbasis = Basis(self.mesh, ElementVector(ElementQuad1()))

    def elem_field(self, img, values):
        """values[phase] per element, broadcast to quadrature points."""
        ph = img[self.pix[0], self.pix[1]]
        v = np.where(ph, values[1], values[0])
        nq = self.basis.X.shape[1]
        return np.repeat(v[:, None], nq, axis=1)

def solve_periodic(K, b, P):
    Kr = (P.T @ K @ P).tocsr(); br = P.T @ b
    # pin one dof (constant mode)
    Kr = Kr[1:, 1:]; br = br[1:]
    w = np.zeros(P.shape[1]); w[1:] = spsolve(Kr.tocsc(), br)
    return P @ w

@BilinearForm
def a_cond(u, v, w): return w.k*dot(grad(u), grad(v))

def k_eff(cell, img, k1, k2, return_field=False):
    kq = cell.elem_field(img, (k1, k2))
    K = asm(a_cond, cell.basis, k=kq)
    Keff = np.zeros((2, 2)); fields = []
    for i, G in enumerate(([1.0, 0.0], [0.0, 1.0])):
        G = np.array(G)
        @LinearForm
        def rhs(v, w): return -w.k*(G[0]*grad(v)[0]+G[1]*grad(v)[1])
        b = asm(rhs, cell.basis, k=kq)
        wsol = solve_periodic(K, b, cell.P)
        gw = cell.basis.interpolate(wsol).grad          # (2, nel, nq)
        jx = kq*(G[0]+gw[0]); jy = kq*(G[1]+gw[1])
        dx = cell.basis.dx
        area = dx.sum()
        Keff[0, i] = (jx*dx).sum()/area; Keff[1, i] = (jy*dx).sum()/area
        fields.append((wsol, jx, jy))
    return (Keff, fields) if return_field else Keff

# plane-strain isotropic elasticity
def lame(E, nu):
    lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu)); return lam, mu

@BilinearForm
def a_el(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return 2*w.mu*ddot(eu, ev) + w.lam*trace(eu)*trace(ev)

def C_eff(cell, img, E1, E2, nu=0.3):
    l1, m1 = lame(E1, nu); l2, m2 = lame(E2, nu)
    lq = cell.elem_field(img, (l1, l2)); mq = cell.elem_field(img, (m1, m2))
    K = asm(a_el, cell.vbasis, lam=lq, mu=mq)
    nn = cell.mesh.p.shape[1]
    # vector dofs are interleaved (x0,y0,x1,y1,...) for ElementVector
    Pv = sp.kron(cell.P, sp.eye(2)).tocsr()
    Es = [np.array([[1, 0], [0, 0.]]), np.array([[0, 0], [0, 1.]]), np.array([[0, .5], [.5, 0]])]
    C = np.zeros((3, 3)); dx = cell.vbasis.dx; area = dx.sum(); stress_xx = None
    for i, Em in enumerate(Es):
        @LinearForm
        def rhs(v, w):
            ev = sym_grad(v)
            return -(2*w.mu*ddot(Em, ev) + w.lam*np.trace(Em)*trace(ev))
        b = asm(rhs, cell.vbasis, lam=lq, mu=mq)
        wsol = solve_periodic(K, b, Pv)
        gw = cell.vbasis.interpolate(wsol).grad        # (2,2,nel,nq)
        eps = 0.5*(gw+np.swapaxes(gw, 0, 1)) + Em[:, :, None, None]
        tr = eps[0, 0]+eps[1, 1]
        sxx = 2*mq*eps[0, 0]+lq*tr; syy = 2*mq*eps[1, 1]+lq*tr; sxy = 2*mq*eps[0, 1]
        C[0, i] = (sxx*dx).sum()/area; C[1, i] = (syy*dx).sum()/area; C[2, i] = (sxy*dx).sum()/area
        if i == 0: stress_xx = sxx
    return C, stress_xx

# ---------------------------------------------------------------- bounds (2D)
def bounds_cond(f2, k1, k2):              # k2 > k1 ; f2 = fraction of phase 2
    f1 = 1-f2
    V = f1*k1+f2*k2; R = 1/(f1/k1+f2/k2)
    HSu = k2 + f1/(1/(k1-k2) + f2/(2*k2))
    HSl = k1 + f2/(1/(k2-k1) + f1/(2*k1))
    return V, R, HSl, HSu

def bounds_bulk2d(f2, K1, G1, K2, G2):    # plane-strain bulk modulus kappa = lam+mu
    f1 = 1-f2
    V = f1*K1+f2*K2; R = 1/(f1/K1+f2/K2)
    HSu = K2 + f1/(1/(K1-K2) + f2/(K2+G2))
    HSl = K1 + f2/(1/(K2-K1) + f1/(K1+G1))
    return V, R, HSl, HSu

if __name__ == "__main__":
    t0 = time.time(); out = {}
    n = 128; k1, k2 = 1.0, 10.0
    # --- 0. validation on a laminate (exact: series/parallel)
    lam_img = np.zeros((n, n), bool); lam_img[n//2:, :] = True    # stripes normal to x
    cell = PeriodicCell(n)
    Kl = k_eff(cell, lam_img, k1, k2)
    V, R, _, _ = bounds_cond(0.5, k1, k2)
    print("[0] laminate check (f2=0.5, k2/k1=10)")
    print(f"    k_xx FEM = {Kl[0,0]:.6f}   Reuss (series)   = {R:.6f}")
    print(f"    k_yy FEM = {Kl[1,1]:.6f}   Voigt (parallel) = {V:.6f}")

    # --- 1. CH microstructures at several mean compositions
    means = [-0.45, -0.3, -0.15, 0.0]
    imgs = {}
    for m in means:
        c = cahn_hilliard(N=n, mean=m, seed=1)
        imgs[m] = c > 0
    np.savez("../../ch05/labs/55_homog_imgs.npz", **{f"m{m:+.2f}": imgs[m] for m in means})
    print("\n[1] CH microstructures (128^2, L=2pi, eps=0.03, dt=0.02, 800 steps, S=2, seed=1)")
    rows = []
    for m in means:
        for swap in (False, True):
            img = ~imgs[m] if swap else imgs[m]
            f2 = img.mean()
            K = k_eff(cell, img, k1, k2)
            kiso = 0.5*np.trace(K); aniso = abs(K[0,0]-K[1,1])/kiso
            V, R, HSl, HSu = bounds_cond(f2, k1, k2)
            rows.append((f2, kiso, aniso, V, R, HSl, HSu, K[0,1]))
    rows.sort()
    print("   f2     k_eff   aniso   Reuss   HS_low  HS_up   Voigt   inside_HS")
    for r in rows:
        f2, kiso, an, V, R, HSl, HSu, kxy = r
        print(f"  {f2:.3f}  {kiso:6.3f}  {an:5.3f}  {R:6.3f}  {HSl:6.3f}  {HSu:6.3f}  {V:6.3f}   {HSl<=kiso<=HSu}")
    out["cond_rows"] = rows

    # --- 2. Keller-Dykhne duality and the symmetric-composition ensemble
    img0 = imgs[0.0]
    print("\n[2] symmetric quench (mean=0): 8 seeds, k1=1, k2=10")
    print("  seed   f2     k_xx    k_yy    k_iso   sqrt(detK(1,10)*detK(10,1))")
    kis = []
    for seed in range(1, 9):
        img = cahn_hilliard(N=n, mean=0.0, seed=seed) > 0
        Ka = k_eff(cell, img, k1, k2); Kb = k_eff(cell, img, k2, k1)
        kiso = np.sqrt(np.linalg.det(Ka)); kis.append(kiso)
        print(f"  {seed:3d}  {img.mean():.3f}  {Ka[0,0]:6.3f}  {Ka[1,1]:6.3f}  {kiso:6.3f}   "
              f"{np.sqrt(np.linalg.det(Ka)*np.linalg.det(Kb)):.4f}")
    kis = np.array(kis); out["kis"] = kis.tolist()
    print(f"  ensemble k_iso = {kis.mean():.3f} +/- {kis.std(ddof=1)/np.sqrt(len(kis)):.3f} (sem), "
          f"sample std = {kis.std(ddof=1):.3f};  Keller-Dykhne sqrt(k1 k2) = {np.sqrt(k1*k2):.4f}")
    for r in (1, 2, 4):
        cr = PeriodicCell(n, r)
        Ka = k_eff(cr, img0, k1, k2); Kb = k_eff(cr, img0, k2, k1)
        print(f"  duality product (seed 1) with {r}x{r} refinement: "
              f"{np.sqrt(np.linalg.det(Ka)*np.linalg.det(Kb)):.4f}  (exact 10)")

    # --- 3. variational upper bound: refinement on a 32x32-pixel crop of the symmetric map (fixed geometry)
    crop = img0[::4, ::4]                     # 32x32 coarse pixel map (same pattern, coarser)
    print("\n[3] mesh refinement at fixed geometry (32^2-pixel map, k2/k1=10)")
    print("   r  elements    k_xx        energy-type quantity decreases monotonically")
    ref = []
    for r in (1, 2, 4, 8):
        c32 = PeriodicCell(32, r)
        K = k_eff(c32, crop, k1, k2)
        ref.append((r, (32*r)**2, K[0,0]))
        print(f"  {r:2d}  {(32*r)**2:7d}   {K[0,0]:.6f}")
    out["refine"] = ref

    # --- 4. plane-strain elasticity, symmetric and droplet maps
    E1, E2, nu = 1.0, 10.0, 0.3
    l1, G1 = lame(E1, nu); l2, G2 = lame(E2, nu); K1, K2 = l1+G1, l2+G2
    print(f"\n[4] plane-strain elasticity (E2/E1=10, nu=0.3): kappa1={K1:.4f}, kappa2={K2:.4f}")
    print("   f2     kappa_eff  Reuss   HS_low  HS_up   Voigt   inside_HS")
    el = []
    for m in means:
        for swap in (False, True):
            img = ~imgs[m] if swap else imgs[m]
            f2 = img.mean()
            C, sxx = C_eff(cell, img, E1, E2, nu)
            kap = 0.25*(C[0,0]+C[1,1]+C[0,1]+C[1,0])
            V, R, HSl, HSu = bounds_bulk2d(f2, K1, G1, K2, G2)
            el.append((f2, kap, V, R, HSl, HSu))
            if m == 0.0 and not swap:
                np.save("../../ch05/labs/55_sxx.npy", sxx.mean(axis=1))
                Csym = C
    el.sort()
    for f2, kap, V, R, HSl, HSu in el:
        print(f"  {f2:.3f}  {kap:7.4f}   {R:6.4f}  {HSl:6.4f}  {HSu:6.4f}  {V:6.4f}   {HSl<=kap<=HSu}")
    print("   C_eff (mean=0 map) =\n", np.array2string(Csym, precision=4, suppress_small=True))
    out["el_rows"] = el

    # save field for figure
    Kf, fields = k_eff(cell, img0, k1, k2, return_field=True)
    jx, jy = fields[0][1].mean(axis=1), fields[0][2].mean(axis=1)
    np.savez("../../ch05/labs/55_flux.npz", jx=jx, jy=jy, pix0=cell.pix[0], pix1=cell.pix[1])
    json.dump(out, open("../../ch05/labs/55_homog.json", "w"))
    print(f"\nruntime {time.time()-t0:.1f} s")
