"""Ex 5.5.4: kinematic uniform BC (w = 0 on the cell boundary) vs periodic BC for the seed-1 CH map.
Also: tiled cells (2x2, 3x3 copies of the same periodic map) to show the boundary-layer effect shrink."""
import sys, time, numpy as np
from scipy.sparse.linalg import spsolve
sys.path.insert(0, "../../../ch05/labs")
import importlib; H = importlib.import_module("55_homog")
from skfem import LinearForm, asm
from skfem.helpers import grad
img = np.load("../../../ch05/labs/55_homog_imgs.npz")["m+0.00"]
k1, k2 = 1.0, 10.0

def k_kubc(cell, img, G):
    kq = cell.elem_field(img, (k1, k2)); K = asm(H.a_cond, cell.basis, k=kq)
    @LinearForm
    def rhs(v, w): return -w.k*(G[0]*grad(v)[0] + G[1]*grad(v)[1])
    b = asm(rhs, cell.basis, k=kq)
    bd = cell.mesh.boundary_nodes(); free = np.setdiff1d(np.arange(K.shape[0]), bd)
    wsol = np.zeros(K.shape[0]); wsol[free] = spsolve(K[free][:, free].tocsc(), b[free])
    gw = cell.basis.interpolate(wsol).grad; dx = cell.basis.dx; A = dx.sum()
    jx = kq*(G[0]+gw[0]); jy = kq*(G[1]+gw[1])
    energy = ((kq*((G[0]+gw[0])**2 + (G[1]+gw[1])**2))*dx).sum()/A
    return energy, (jx*dx).sum()/A, (jy*dx).sum()/A, (gw[0]*dx).sum()/A

t0 = time.time()
print(f"seed-1 map: f2={img.mean():.3f}")
for rep in (1, 2, 3):
    big = np.tile(img, (rep, rep)); n = big.shape[0]
    cell = H.PeriodicCell(n)
    Kp = H.k_eff(cell, big, k1, k2)
    eK, jx, jy, mg = k_kubc(cell, big, (1.0, 0.0))
    eKy, _, _, _ = k_kubc(cell, big, (0.0, 1.0))
    print(f"{rep}x{rep} tiling ({n}^2 Q1): periodic K_xx={Kp[0,0]:.4f} K_yy={Kp[1,1]:.4f} | KUBC K_xx={eK:.4f} "
          f"(<j_x>={jx:.4f}, <dw/dx>={mg:.1e}) K_yy={eKy:.4f} | excess xx {eK/Kp[0,0]-1:+.3%}  ({time.time()-t0:.0f} s)", flush=True)
