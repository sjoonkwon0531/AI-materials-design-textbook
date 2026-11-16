"""Cross-check for lab_5_8.jl: same tables as the Julia file (OPP written 1-based there, 0-based here),
pull streaming fn[i,x,y] = fpost[i, x-cx, y-cy] (mod), halfway bounce-back when the source is solid,
macros with rho=1 on solid nodes, convergence test every 200 steps. Vectorized numpy."""
import numpy as np
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1]); CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
W9 = np.array([4/9] + [1/9]*4 + [1/36]*4); OPP1 = [1, 4, 5, 2, 3, 8, 9, 6, 7]; OPP = [o-1 for o in OPP1]
def step(f, tau, Fx, Fy, solid):
    rho = f.sum(0); fl = ~solid; rr = np.where(fl, rho, 1.0)
    ux = ((f*CX[:, None, None]).sum(0) + 0.5*Fx)/rr; uy = ((f*CY[:, None, None]).sum(0) + 0.5*Fy)/rr
    cu = CX[:, None, None]*ux + CY[:, None, None]*uy; usq = ux**2 + uy**2
    fe = W9[:, None, None]*rho*(1 + 3*cu + 4.5*cu**2 - 1.5*usq)
    Si = W9[:, None, None]*((3*(CX[:, None, None]-ux) + 9*cu*CX[:, None, None])*Fx + (3*(CY[:, None, None]-uy) + 9*cu*CY[:, None, None])*Fy)
    fp = np.where(fl, f - (f-fe)/tau + (1-0.5/tau)*Si, 0.0)
    fn = np.zeros_like(f)
    for i in range(9):
        src = np.roll(np.roll(fp[i], CX[i], 0), CY[i], 1); srcsolid = np.roll(np.roll(solid, CX[i], 0), CY[i], 1)
        fn[i] = np.where(fl, np.where(srcsolid, fp[OPP[i]], src), 0.0)
    return fn
def pois(ny, tau, g, nx=3, tol=1e-12):
    Ny = ny+2; solid = np.zeros((nx, Ny), bool); solid[:, 0] = solid[:, -1] = True
    Fx = np.where(solid, 0, g); Fy = np.zeros_like(Fx); rho = np.where(solid, 0., 1.)
    f = W9[:, None, None]*rho[None]; m0 = f.sum(); uold = None; drift = 0
    for it in range(1, 400001):
        f = step(f, tau, Fx, Fy, solid)
        if it % 200 == 0:
            drift = max(drift, abs(f.sum()-m0)/m0)
            rr = np.where(f.sum(0) > 0, f.sum(0), 1.0); u = ((f*CX[:, None, None]).sum(0)/rr + 0.5*Fx/rr)[1, 1:-1]
            if uold is not None and np.max(np.abs(u-uold)) < tol*np.max(np.abs(u)): break
            uold = u
    nu = (tau-0.5)/3; y = np.arange(1, ny+1)-0.5; ua = g/(2*nu)*y*(ny-y)
    return np.sqrt(((u-ua)**2).sum()/(ua**2).sum()), it, drift, np.mean(u-ua)*nu/g
for ny in (8, 16):
    tau = 0.8; nu = (tau-0.5)/3; g = 0.01*8*nu/ny**2
    e, it, dr, sl = pois(ny, tau, g)
    print(f"ny={ny} tau=0.8: err={e:.3e} iters={it} drift={dr:.1e} slip*nu/g={sl:+.5f}")
print("chapter: ny=8 err 1.112e-02; ny=16 err 2.781e-03, iters 7200, drift 4.7e-13, slip -0.06500")
