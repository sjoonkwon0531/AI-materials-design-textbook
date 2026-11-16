"""Lab 55_lbm -- D2Q9 BGK lattice Boltzmann: body-force-driven Poiseuille flow (validation)
and a cylinder in a channel (demonstration). Lattice units (dx = dt = 1).
Guo forcing, halfway bounce-back walls. Deterministic. Runtime ~1 min.
"""
import numpy as np, time

# D2Q9 lattice
c = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
w = np.array([4/9] + [1/9]*4 + [1/36]*4)
opp = np.array([0,3,4,1,2,7,8,5,6])
cs2 = 1/3

def feq(rho, ux, uy):
    cu = c[:,0,None,None]*ux + c[:,1,None,None]*uy
    usq = ux**2 + uy**2
    return w[:,None,None]*rho*(1 + cu/cs2 + 0.5*cu**2/cs2**2 - 0.5*usq/cs2)

def guo_source(ux, uy, Fx, Fy, tau):
    cu = c[:,0,None,None]*ux + c[:,1,None,None]*uy
    S = np.zeros((9,)+ux.shape)
    for i in range(9):
        ex, ey = c[i]
        S[i] = w[i]*(((ex-ux)/cs2 + cu[i]*ex/cs2**2)*Fx + ((ey-uy)/cs2 + cu[i]*ey/cs2**2)*Fy)
    return (1 - 0.5/tau)*S

def macros(f, Fx, Fy):
    rho = f.sum(0)
    rho = np.where(rho > 0, rho, 1.0)          # solid nodes carry no populations
    ux = (f*c[:,0,None,None]).sum(0)/rho + 0.5*Fx/rho
    uy = (f*c[:,1,None,None]).sum(0)/rho + 0.5*Fy/rho
    return rho, ux, uy

def step(f, tau, Fx, Fy, solid):
    rho, ux, uy = macros(f, Fx, Fy)
    fpost = f - (f - feq(rho, ux, uy))/tau + guo_source(ux, uy, Fx, Fy, tau)
    # streaming (periodic), then halfway bounce-back: populations that entered a solid node
    fs = np.empty_like(fpost)
    for i in range(9):
        fs[i] = np.roll(np.roll(fpost[i], c[i,0], axis=0), c[i,1], axis=1)
    # halfway bounce-back: any population that streamed into a solid node is reflected back
    # to the fluid node it came from, with reversed direction, within the same step
    for i in range(9):
        into_solid = np.roll(np.roll(solid, -c[i,0], axis=0), -c[i,1], axis=1) & ~solid
        fs[opp[i]][into_solid] = fpost[i][into_solid]
    fs[:, solid] = 0.0
    return fs

from numba import njit

@njit(cache=True)
def step_numba(f, tau, Fx, Fy, solid):
    """Same algorithm as step(), fused into explicit loops (pull streaming)."""
    nx, ny = solid.shape
    fpost = np.zeros_like(f)
    for x in range(nx):
        for y in range(ny):
            if solid[x, y]:
                continue
            rho = 0.0; mx = 0.0; my = 0.0
            for i in range(9):
                rho += f[i, x, y]; mx += f[i, x, y]*c[i, 0]; my += f[i, x, y]*c[i, 1]
            ux = (mx + 0.5*Fx[x, y])/rho; uy = (my + 0.5*Fy[x, y])/rho
            usq = ux*ux + uy*uy
            for i in range(9):
                cu = c[i, 0]*ux + c[i, 1]*uy
                fe = w[i]*rho*(1 + 3*cu + 4.5*cu*cu - 1.5*usq)
                Si = w[i]*((3*(c[i, 0]-ux) + 9*cu*c[i, 0])*Fx[x, y] + (3*(c[i, 1]-uy) + 9*cu*c[i, 1])*Fy[x, y])
                fpost[i, x, y] = f[i, x, y] - (f[i, x, y]-fe)/tau + (1-0.5/tau)*Si
    fn = np.zeros_like(f)
    for x in range(nx):
        for y in range(ny):
            if solid[x, y]:
                continue
            for i in range(9):
                xs = (x - c[i, 0]) % nx; ys = (y - c[i, 1]) % ny
                if solid[xs, ys]:
                    fn[i, x, y] = fpost[opp[i], x, y]     # halfway bounce-back
                else:
                    fn[i, x, y] = fpost[i, xs, ys]
    return fn

def poiseuille(ny, tau, g, tol=1e-12, maxit=400000, nx=3):
    """Channel: fluid rows y=1..ny, solid rows y=0 and y=ny+1; walls located halfway (y=0.5, ny+0.5)."""
    Ny = ny + 2
    solid = np.zeros((nx, Ny), bool); solid[:, 0] = solid[:, -1] = True
    Fx = np.where(solid, 0.0, g); Fy = np.zeros_like(Fx)
    f = feq(np.where(solid, 0, 1.0), np.zeros((nx, Ny)), np.zeros((nx, Ny)))
    m0 = f.sum(); uold = None; drift = 0.0
    for it in range(1, maxit+1):
        f = step_numba(f, tau, Fx, Fy, solid)
        if it % 200 == 0:
            drift = max(drift, abs(f.sum()-m0)/m0)
            _, ux, _ = macros(f, Fx, Fy)
            u = ux[1, 1:-1].copy()
            if uold is not None and np.max(np.abs(u-uold)) < tol*np.max(np.abs(u)):
                break
            uold = u
    nu = cs2*(tau-0.5)
    y = np.arange(1, ny+1) - 0.5           # distance from wall at y=0.5
    H = ny
    ua = g/(2*nu)*y*(H-y)
    err = np.sqrt(np.sum((u-ua)**2)/np.sum(ua**2))
    return y, u, ua, err, it, drift

if __name__ == "__main__":
    t0 = time.time()
    # consistency check: numpy step() and numba step_numba() give identical results
    nx_, ny_ = 5, 8; sol = np.zeros((nx_, ny_+2), bool); sol[:, 0] = sol[:, -1] = True
    Fx_ = np.where(sol, 0, 1e-4); Fy_ = np.zeros_like(Fx_)
    rng = np.random.default_rng(0)
    f_ = feq(np.where(sol, 0, 1.0), 0.01*rng.standard_normal(sol.shape), 0.01*rng.standard_normal(sol.shape))
    f_[:, sol] = 0
    a_, b_ = f_.copy(), f_.copy()
    for _ in range(50):
        a_ = step(a_, 0.8, Fx_, Fy_, sol); b_ = step_numba(b_, 0.8, Fx_, Fy_, sol)
    print(f"[0] numpy vs numba kernel, 50 steps: max |diff| = {np.abs(a_-b_).max():.1e}")
    print("[1] Poiseuille, ny=16, g chosen so that u_max = 0.01 (Ma ~ 0.017)")
    print("   tau     nu       L2 rel err    iters   max mass drift   slip*nu/g   (16tau^2-16tau+1)/24")
    taus = [0.55, 0.6, 0.7, 0.8, 0.9, 0.933, 1.0, 1.2, 1.5, 2.0]
    res_tau = []
    for tau in taus:
        nu = cs2*(tau-0.5); ny = 16; g = 0.01*8*nu/ny**2
        y, u, ua, err, it, dr = poiseuille(ny, tau, g)
        res_tau.append((tau, err))
        slip = np.mean(u-ua)*nu/g          # u - ua is a uniform shift (std/mean < 1e-9)
        print(f"  {tau:5.3f}  {nu:.4f}   {err:.3e}    {it:6d}   {dr:.1e}       {slip:+.5f}    {(16*tau**2-16*tau+1)/24:+.5f}")
    np.save("../../ch05/labs/55_lbm_tau.npy", np.array(res_tau))

    print("\n[2] grid convergence at tau=0.8 (u_max=0.01 fixed, Ma fixed)")
    print("   ny     L2 rel err    order")
    res_n = []; prev = None
    for ny in (8, 16, 32, 64):
        tau = 0.8; nu = cs2*(tau-0.5); g = 0.01*8*nu/ny**2
        y, u, ua, err, it, dr = poiseuille(ny, tau, g)
        order = "" if prev is None else f"{np.log2(prev/err):.2f}"
        print(f"  {ny:3d}    {err:.3e}    {order}")
        res_n.append((ny, err)); prev = err
        if ny == 16: np.savez("../../ch05/labs/55_lbm_prof.npz", y=y, u=u, ua=ua)
    np.save("../../ch05/labs/55_lbm_n.npy", np.array(res_n))

    print("\n[3] cylinder in a periodic channel (body-force driven), two forcing levels")
    nx, ny, R, tau = 200, 60, 8, 0.6
    Ny = ny + 2; nu = cs2*(tau-0.5)
    X, Y = np.meshgrid(np.arange(nx), np.arange(Ny), indexing='ij')
    solid = (Y == 0) | (Y == Ny-1) | ((X-nx//4)**2 + (Y-Ny/2+0.5)**2 < R**2)
    print(f"   grid {nx}x{ny}, D={2*R}, tau={tau}, nu={nu:.4f}, 25000 steps each")
    for g, tag in ((2.0e-6, "lowRe"), (2.0e-5, "highRe")):
        Fx = np.where(solid, 0.0, g); Fy = np.zeros_like(Fx)
        f = feq(np.where(solid, 0, 1.0), np.zeros((nx, Ny)), np.zeros((nx, Ny)))
        m0 = f.sum()
        for it in range(25000):
            f = step_numba(f, tau, Fx, Fy, solid)
        rho, ux, uy = macros(f, Fx, Fy)
        ux[solid] = 0; uy[solid] = 0
        U = ux[~solid].mean(); Re = U*2*R/nu
        umax = np.sqrt(ux**2+uy**2).max()
        yc = int(round(Ny/2-0.5)); xs = np.arange(nx//4+R, nx)
        rev = xs[ux[xs, yc] < 0]
        L_rec = (rev.max() - (nx//4+R) + 1) if rev.size else 0
        print(f"   g={g:.1e}: mean u={U:.3e}, max|u|={umax:.3e} (Ma={umax/np.sqrt(cs2):.3f}), Re_D={Re:.1f}, "
              f"mass drift={abs(f.sum()-m0)/m0:.1e}, recirculation length={L_rec} lu ({L_rec/(2*R):.2f} D)")
        np.savez(f"../../ch05/labs/55_lbm_cyl_{tag}.npz", ux=ux, uy=uy, solid=solid, Re=Re)
    print(f"\nruntime {time.time()-t0:.1f} s")
