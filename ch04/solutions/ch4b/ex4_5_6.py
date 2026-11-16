# Ex 4.5.6: breaking rotational invariance -> (A) conservative but orientation-dependent energy: E conserved, L not
#                                            (B) direct (non-gradient) force model with lab-frame angle: E not conserved
import numpy as np
rc = 2.5; a = 0.2                              # a: strength of lab-frame (absolute-orientation) dependence
def U(r):  return 4*(r**-12-r**-6) - 4*(rc**-12-rc**-6)
def Fr(r): return 48*r**-13 - 24*r**-7          # -dU/dr
def pairs(pos, L):
    d = pos[:,None,:]-pos[None,:,:]
    if L: d -= L*np.round(d/L)
    r = np.sqrt((d**2).sum(-1)); np.fill_diagonal(r, np.inf); return d, r
def model(pos, L, kind):
    d, r = pairs(pos, L); m = r < rc; rr = np.where(m, r, 1.0)
    dx, dy = d[...,0], d[...,1]; c2 = (dx**2-dy**2)/rr**2           # cos(2 phi), phi = bond angle in LAB frame
    u = np.where(m, U(rr), 0.0); fr = np.where(m, Fr(rr), 0.0)
    if kind == "true": w = 1.0; E = 0.5*u.sum()
    else:              w = 1 + a*c2; E = 0.5*(u*w).sum()
    # radial part:  -dU/dr * w * d/r
    F = ((fr*w/rr)[...,None]*d)
    if kind == "A":   # exact gradient of E_A  -> extra tangential term from d(cos2phi)/d(d)
        gx = a*u*(2*dx/rr**2 - (dx**2-dy**2)*2*dx/rr**4)
        gy = a*u*(-2*dy/rr**2 - (dx**2-dy**2)*2*dy/rr**4)
        F -= np.stack([gx, gy], -1)
    # kind "B": direct force model = radial force only (what a force-predicting net fed lab-frame angles may learn)
    return F.sum(1), E
def nve(pos, vel, L, kind, nsteps, dt):
    F, E = model(pos, L, kind); out = []
    for s in range(nsteps+1):
        if s % 500 == 0:
            KE = 0.5*(vel**2).sum(); com = pos.mean(0) if not L else 0
            Lz = np.sum((pos-com)[:,0]*vel[:,1]-(pos-com)[:,1]*vel[:,0]) if not L else np.nan
            out.append((s, KE+E, KE, Lz))
        vel += 0.5*dt*F; pos = pos+dt*vel
        if L: pos %= L
        F, E = model(pos, L, kind); vel += 0.5*dt*F
    return np.array(out)
# --- (1) periodic 2D liquid, NVE (Lab 4.4 conditions: N=64, rho=0.7, T=0.7) ---
N=64; L=np.sqrt(N/0.7); xs=(np.arange(8)+0.5)*L/8; X,Y=np.meshgrid(xs,xs); p0=np.stack([X.ravel(),Y.ravel()],1)
rng=np.random.default_rng(0); v0=rng.standard_normal((N,2)); v0-=v0.mean(0); v0*=np.sqrt(0.7/((v0**2).sum()/(2*N-2)))
print("(1) PBC liquid NVE, dt=0.002, 20000 steps; total energy per atom (KE + model PE)")
for kind in ["true","A","B"]:
    o = nve(p0.copy(), v0.copy(), L, kind, 20000, 0.002)
    print(f"  {kind:4s}: E/N start={o[0,1]/N:+.4f}  end={o[-1,1]/N:+.4f}  drift={(o[-1,1]-o[0,1])/N:+.4f}  "
          f"max|E-E0|/N={np.abs(o[:,1]-o[0,1]).max()/N:.4f}  T_end={o[-1,2]/(N-1):.3f}")
    print("        E/N every 5000 steps:", np.round(o[::10,1]/N,4))
# --- (2) isolated 2D cluster (no PBC), zero initial angular momentum ---
pts=[(0,0)]+[(np.cos(k*np.pi/3),np.sin(k*np.pi/3)) for k in range(6)]
pts+= [(2*np.cos(k*np.pi/3),2*np.sin(k*np.pi/3)) for k in range(6)]+[(np.sqrt(3)*np.cos(k*np.pi/3+np.pi/6),np.sqrt(3)*np.sin(k*np.pi/3+np.pi/6)) for k in range(6)]
pc=np.array(pts)*2**(1/6); Nc=len(pc)
vc=0.2*rng.standard_normal((Nc,2)); vc-=vc.mean(0)
Lz0=np.sum(pc[:,0]*vc[:,1]-pc[:,1]*vc[:,0]); I=np.sum((pc**2).sum(1)); w=Lz0/I
vc-= w*np.stack([-pc[:,1],pc[:,0]],1)          # remove initial rotation -> L_z = 0
print(f"(2) isolated {Nc}-atom cluster NVE, dt=0.002, 20000 steps (L_z(0) = 0)")
for kind in ["true","A","B"]:
    o = nve(pc.copy(), vc.copy(), 0, kind, 20000, 0.002)
    print(f"  {kind:4s}: max|E-E0| = {np.abs(o[:,1]-o[0,1]).max():.4f}   E_end-E0 = {o[-1,1]-o[0,1]:+.4f}   max|L_z| = {np.abs(o[:,3]).max():.4f}")
