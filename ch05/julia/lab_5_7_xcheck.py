"""Cross-check for lab_5_7.jl: the SAME native Q1 assembly (2x2 Gauss reference matrices, periodic
node folding, pinned dofs, element-integrated averages) re-implemented line by line in numpy/scipy,
fed with the chapter's numpy Cahn-Hilliard maps. It must reproduce the chapter's scikit-fem numbers.
Indices are written 0-based here; lab_5_7.jl uses the same formulas 1-based."""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
sys.path.insert(0, "../ch05/labs")
from importlib import import_module
H = import_module("55_homog")            # only cahn_hilliard() and the bounds are reused

XA = (-1., 1., 1., -1.); YA = (-1., -1., 1., 1.)
def q1_reference(h):
    gp = (-1/np.sqrt(3), 1/np.sqrt(3)); detJ = (h/2)**2
    Klap = np.zeros((4, 4)); gint = np.zeros((4, 2)); Klam = np.zeros((8, 8)); Kmu = np.zeros((8, 8)); Bint = np.zeros((3, 8))
    Dl = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0.]]); Dm = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1.]])
    for xi in gp:
        for eta in gp:
            dN = np.zeros((4, 2))
            for a in range(4):
                dN[a, 0] = XA[a]*(1+YA[a]*eta)/4*(2/h); dN[a, 1] = YA[a]*(1+XA[a]*xi)/4*(2/h)
            B = np.zeros((3, 8))
            for a in range(4):
                B[0, 2*a] = dN[a, 0]; B[1, 2*a+1] = dN[a, 1]; B[2, 2*a] = dN[a, 1]; B[2, 2*a+1] = dN[a, 0]
            Klap += dN @ dN.T * detJ; gint += dN*detJ; Klam += B.T @ Dl @ B*detJ; Kmu += B.T @ Dm @ B*detJ; Bint += B*detJ
    return Klap, gint, Klam, Kmu, Bint

def cell(n, r=1):
    m = n*r; conn = []; pix = []
    node = lambda I, J: (I % m)*m + (J % m)
    for ie in range(m):
        for je in range(m):
            conn.append([node(ie, je), node(ie+1, je), node(ie+1, je+1), node(ie, je+1)]); pix.append((ie//r, je//r))
    return m, 1.0/m, np.array(conn), np.array(pix)

def k_eff(c, img, k1, k2):
    m, h, conn, pix = c; Klap, gint, *_ = q1_reference(h)
    kq = np.where(img[pix[:, 0], pix[:, 1]], k2, k1); nd = m*m
    I = np.repeat(conn, 4, axis=1).ravel(); J = np.tile(conn, (1, 4)).ravel()
    V = (kq[:, None, None]*Klap[None]).reshape(len(kq), 16).ravel()
    K = sp.csc_matrix((V, (I, J)), shape=(nd, nd)); lu = splu(K[1:, 1:].tocsc())
    Keff = np.zeros((2, 2))
    for i, G in enumerate(([1., 0.], [0., 1.])):
        fe = -kq[:, None]*(gint @ np.array(G))[None, :]
        rhs = np.zeros(nd); np.add.at(rhs, conn.ravel(), fe.ravel())
        w = np.zeros(nd); w[1:] = lu.solve(rhs[1:])
        gw = w[conn] @ gint                         # (ne, 2) integrated gradient
        Keff[0, i] = (kq*(G[0]*h*h + gw[:, 0])).sum(); Keff[1, i] = (kq*(G[1]*h*h + gw[:, 1])).sum()
    return Keff

def C_eff(c, img, E1, E2, nu=0.3):
    m, h, conn, pix = c; _, _, Klam, Kmu, Bint = q1_reference(h)
    l1, m1 = H.lame(E1, nu); l2, m2 = H.lame(E2, nu); ph = img[pix[:, 0], pix[:, 1]]
    lq = np.where(ph, l2, l1); mq = np.where(ph, m2, m1); nd = 2*m*m
    dofs = np.stack([2*conn[:, a//2] + (a % 2) for a in range(8)], 1)
    I = np.repeat(dofs, 8, axis=1).ravel(); J = np.tile(dofs, (1, 8)).ravel()
    V = (lq[:, None, None]*Klam[None] + mq[:, None, None]*Kmu[None]).reshape(len(lq), 64).ravel()
    K = sp.csc_matrix((V, (I, J)), shape=(nd, nd)); lu = splu(K[2:, 2:].tocsc())
    Dl = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0.]]); Dm = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1.]])
    D = lq[:, None, None]*Dl + mq[:, None, None]*Dm
    C = np.zeros((3, 3))
    for i in range(3):
        Ev = np.zeros(3); Ev[i] = 1.0
        fe = -np.einsum('ka,ekl,l->ea', Bint, D, Ev)
        rhs = np.zeros(nd); np.add.at(rhs, dofs.ravel(), fe.ravel())
        w = np.zeros(nd); w[2:] = lu.solve(rhs[2:])
        strain = Ev[None]*h*h + w[dofs] @ Bint.T
        C[:, i] = np.einsum('ekl,el->k', D, strain)
    return C

n = 128; c1 = cell(n)
lam = np.zeros((n, n), bool); lam[n//2:, :] = True
Kl = k_eff(c1, lam, 1., 10.)
print(f"[0] k_xx={Kl[0,0]:.6f} (chapter 1.818182)  k_yy={Kl[1,1]:.6f} (chapter 5.500000)")
img0 = H.cahn_hilliard(N=n, mean=0.0, seed=1) > 0
imgd = H.cahn_hilliard(N=n, mean=-0.45, seed=1) > 0
K = k_eff(c1, imgd, 1., 10.); print(f"[1] f2={imgd.mean():.3f} k_eff={0.5*np.trace(K):.3f} aniso={abs(K[0,0]-K[1,1])/(0.5*np.trace(K)):.3f} (chapter 0.261 1.584 0.044)")
Ka = k_eff(c1, img0, 1., 10.); Kb = k_eff(c1, img0, 10., 1.)
print(f"[2] seed1 f2={img0.mean():.3f} kxx={Ka[0,0]:.3f} kyy={Ka[1,1]:.3f} kiso={np.sqrt(np.linalg.det(Ka)):.3f} dual={np.sqrt(np.linalg.det(Ka)*np.linalg.det(Kb)):.4f} (chapter 0.497 2.690 2.789 2.732 10.1122)")
crop = img0[::4, ::4]
print("[3] r=1,2:", [f"{k_eff(cell(32, r), crop, 1., 10.)[0,0]:.6f}" for r in (1, 2)], "(chapter 2.785884, 2.750014)")
C = C_eff(c1, img0, 1., 10., 0.3)
print(f"[4] kappa_eff(mean=0)={0.25*(C[0,0]+C[1,1]+C[0,1]+C[1,0]):.4f} (chapter 2.1482)")
print("    C_eff =", np.array2string(C, precision=4, suppress_small=True).replace("\n", " "),
      "(chapter [[3.1232 1.1413 -0.0579] [1.1413 3.1871 -0.0724] [-0.0579 -0.0724 0.8987]])")
