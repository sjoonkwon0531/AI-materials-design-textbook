"""Verification code for Ch.4 exercises 4.4.1-4.4.6 (MD, Lab 4.4)."""
import time
import numpy as np
import sympy as sp

# =========================== 4.4.1 time reversibility ===========================
print("=== 4.4.1 symbolic: one VV step then dt -> -dt returns the initial state (SHO and general F) ===")
x0, v0, dt = sp.symbols('x0 v0 dt', real=True)
F = sp.Function('F')            # arbitrary position-dependent force, m=1


def vv_step(x, v, h):
    v_half = v + h/2*F(x)
    x_new = x + h*v_half
    v_new = v_half + h/2*F(x_new)
    return x_new, v_new


x1, v1 = vv_step(x0, v0, dt)
x2, v2 = vv_step(x1, v1, -dt)
print("  x2 - x0 =", sp.simplify(x2 - x0), ";  v2 - v0 =", sp.simplify(v2 - v0))
# velocity-flip version: (x1, -v1) with +dt gives (x0, -v0)
x3, v3 = vv_step(x1, -v1, dt)
print("  flip version: x3 - x0 =", sp.simplify(x3 - x0), ";  v3 + v0 =", sp.simplify(v3 + v0))
# forward Euler is NOT reversible
xe1 = x0 + dt*v0; ve1 = v0 + dt*F(x0)
xe2 = xe1 - dt*ve1
print("  forward Euler: x2 - x0 =", sp.simplify(xe2 - x0), " (nonzero -> irreversible)")

# =========================== 4.4.2 LJ minimum ===========================
print("\n=== 4.4.2 LJ ===")
r, eps_, sig = sp.symbols('r epsilon sigma', positive=True)
U = 4*eps_*((sig/r)**12 - (sig/r)**6)
crit = sp.solve(sp.Eq(sp.diff(U, r), 0), r)
print("  dU/dr=0 ->", crit, "; U(rmin) =", sp.simplify(U.subs(r, crit[0])), "; U(sigma) =", U.subs(r, sig))
k_LJ = sp.simplify(sp.diff(U, r, 2).subs(r, crit[0]))
print("  U''(rmin) =", k_LJ, "=", float(k_LJ.subs({eps_: 1, sig: 1})), "(eps/sigma^2)")

# =========================== 4.4.3 SHO dt dependence ===========================
print("\n=== 4.4.3 SHO: Verlet vs forward Euler vs dt (same physical time T=45) ===")


def sho_run(method, dt, nsteps, x0=1.0, v0=0.0):
    x, v = x0, v0; E = [0.5*(v*v + x*x)]
    for _ in range(nsteps):
        if method == "verlet":
            v += 0.5*dt*(-x); x += dt*v; v += 0.5*dt*(-x)
        elif method == "euler":
            xn = x + dt*v; v = v + dt*(-x); x = xn
        E.append(0.5*(v*v + x*x))
    return np.array(E)


Ttot = 45.0
amps = []
dts = [0.05, 0.15, 0.3]
for d in dts:
    n = int(round(Ttot/d))
    Ev = sho_run("verlet", d, n); Ee = sho_run("euler", d, n)
    amp = (Ev.max() - Ev.min())/0.5; amps.append(amp)
    print(f"  dt={d:4.2f} ({n:4d} steps): Euler E_end/E0={Ee[-1]/0.5:10.4e} "
          f"(theory (1+dt^2)^n={(1+d*d)**n:10.4e}) | Verlet peak-to-peak/E0={amp:.3e}, "
          f"theory dt^2/4={d*d/4:.3e}")
# shadow invariant of velocity Verlet for the SHO:  I = v^2/2 + (1 - dt^2/4) x^2/2
for d in dts:
    x, v = 1.0, 0.0; Is = []
    for _ in range(int(round(Ttot/d))):
        v += 0.5*d*(-x); x += d*v; v += 0.5*d*(-x)
        Is.append(0.5*v*v + 0.5*(1 - d*d/4)*x*x)
    print(f"  dt={d}: shadow invariant I=v^2/2+(1-dt^2/4)x^2/2: max|I-I0|/I0={np.max(np.abs(np.array(Is)/(0.5*(1-d*d/4))-1)):.1e}")
for i in range(1, len(dts)):
    print(f"  Verlet amplitude order between dt={dts[i-1]} and {dts[i]}: "
          f"{np.log(amps[i]/amps[i-1])/np.log(dts[i]/dts[i-1]):.3f}")
# same number of steps (300) as in Lab Part A
for d in dts:
    Ee = sho_run("euler", d, 300)
    print(f"  300 steps, dt={d}: Euler E_end={Ee[-1]:.4e}")

# =========================== LJ engine (vectorised Lab 4.4) ===========================
rc = 2.5; rc2 = rc*rc; ushift = 4.0*(rc**-12 - rc**-6)


def init_lattice(nside, density):
    N = nside*nside; L = np.sqrt(N/density); a = L/nside
    xs = (np.arange(nside) + 0.5)*a; X, Y = np.meshgrid(xs, xs)
    return np.stack([X.ravel(), Y.ravel()], axis=1), L, N


def forces_energy(pos, L):
    """Same physics as Lab 4.4 forces_energy (cutoff 2.5, shifted energy, minimum image), vectorised."""
    rij = pos[:, None, :] - pos[None, :, :]
    rij -= L*np.round(rij/L)
    r2 = np.sum(rij*rij, axis=2)
    np.fill_diagonal(r2, np.inf)
    m = r2 < rc2
    inv2 = np.where(m, 1.0/r2, 0.0); inv6 = inv2**3
    fs = (48.0*inv6*inv6 - 24.0*inv6)*inv2
    F = np.sum(rij*fs[:, :, None], axis=1)
    PE = 0.5*np.sum(np.where(m, 4.0*(inv6*inv6 - inv6) - ushift, 0.0))
    return F, PE


def kinetic_temp(vel):
    KE = 0.5*np.sum(vel*vel); N = len(vel); return KE, 2.0*KE/(2*N - 2)


def nve_run(nsteps, dt, seed=1, nside=6, density=0.7, T0=1.0, every=5):
    """Lab 4.4 Part B: identical initial condition (seed 1 is the first random draw in lab_4_4.py)."""
    rng = np.random.RandomState(seed)
    pos, L, N = init_lattice(nside, density)
    vel = rng.randn(N, 2); vel -= vel.mean(axis=0)
    _, T = kinetic_temp(vel); vel *= np.sqrt(T0/T)
    F, PE = forces_energy_lab(pos, L); Es = []
    for s in range(nsteps):
        vel += 0.5*dt*F; pos = (pos + dt*vel) % L; F, PE = forces_energy_lab(pos, L); vel += 0.5*dt*F
        if s % every == 0:
            Es.append(0.5*np.sum(vel*vel) + PE)
    return np.array(Es), pos, vel, L


def forces_energy_lab(pos, L):
    """Verbatim Lab 4.4 loop force routine (used for the dt scan so that dt=0.004 reproduces Lab Part B)."""
    N = len(pos); F = np.zeros_like(pos); PE = 0.0
    for i in range(N-1):
        rij = pos[i]-pos[i+1:]; rij -= L*np.round(rij/L)
        r2 = np.sum(rij*rij, axis=1); m = r2 < rc2
        r2m = r2[m]; rijm = rij[m]; inv2 = 1.0/r2m; inv6 = inv2**3
        fs = (48.0*inv6*inv6-24.0*inv6)*inv2
        fv = rijm*fs[:, None]; F[i] += fv.sum(axis=0)
        np.add.at(F, np.where(m)[0]+(i+1), -fv)
        PE += np.sum(4.0*(inv6*inv6-inv6)-ushift)
    return F, PE


# ---- 4.4.1 numerical reversibility on the LJ system ----
print("\n=== 4.4.1 numerical: LJ N=36, 1000 steps forward, flip v, 1000 steps ===")
rng = np.random.RandomState(1)
pos, L, N = init_lattice(6, 0.7)
vel = rng.randn(N, 2); vel -= vel.mean(axis=0); vel *= np.sqrt(1.0/kinetic_temp(vel)[1])
p0 = pos.copy(); v0_ = vel.copy()
dt_ = 0.004
for nst in [100, 1000]:
    pos = p0.copy(); vel = v0_.copy()
    F, _ = forces_energy(pos, L)
    for s in range(nst):
        vel += 0.5*dt_*F; pos = pos + dt_*vel; F, _ = forces_energy(pos, L); vel += 0.5*dt_*F
    vel = -vel
    for s in range(nst):
        vel += 0.5*dt_*F; pos = pos + dt_*vel; F, _ = forces_energy(pos, L); vel += 0.5*dt_*F
    print(f"  {nst:5d} steps there and back: max|R-R0|={np.max(np.abs(pos-p0)):.2e}, "
          f"max|v+v0|={np.max(np.abs(vel+v0_)):.2e}")

# =========================== 4.4.4 LJ NVE dt scan ===========================
print("\n=== 4.4.4 LJ NVE dt scan (same physical time t=24, same initial condition) ===")
E_chk, *_ = nve_run(6000, 0.004)
print(f"  check vs Lab Part B (dt=0.004, 6000 steps): pp/|E|={(E_chk.max()-E_chk.min())/abs(E_chk.mean()):.2e}, "
      f"E(end)/E(0)={E_chk[-1]/E_chk[0]:.4f}, mean E={E_chk.mean():.3f}")
res = {}
for d in [0.002, 0.004, 0.008, 0.016, 0.024, 0.032, 0.040]:
    n = int(round(24.0/d))
    E, *_ = nve_run(n, d, every=5)
    finite = np.all(np.isfinite(E))
    pp = (E.max() - E.min())/abs(E.mean()) if finite else np.inf
    drift = (E[-1] - E[0])/abs(E[0]) if finite else np.inf
    res[d] = pp
    print(f"  dt={d:5.3f} ({n:5d} steps): peak-to-peak/|E|={pp:.2e}, drift (E_end-E0)/|E0|={drift:+.2e}")
ds = [0.002, 0.004, 0.008, 0.016]
for i in range(1, len(ds)):
    print(f"  local order {ds[i-1]}->{ds[i]}: {np.log(res[ds[i]]/res[ds[i-1]])/np.log(2):.2f}")
pfit = np.polyfit(np.log(ds), np.log([res[d] for d in ds]), 1)[0]
print(f"  least-squares order over dt=0.002..0.016: {pfit:.2f}")
# vibration period estimate
kpair = float(k_LJ.subs({eps_: 1, sig: 1}))
w_pair = np.sqrt(kpair/0.5)
print(f"  LJ dimer: omega=sqrt(U''/mu)={w_pair:.3f}, period={2*np.pi/w_pair:.3f}, period/20={2*np.pi/w_pair/20:.4f}")


# =========================== 4.4.6 AIMD vs MLIP arithmetic ===========================
print("\n=== 4.4.6 ===")
nsteps = 100e-12/1e-15
t_dft = nsteps*30.0; t_ml = nsteps*1e-3
print(f"  steps={nsteps:.0e}; BOMD: {t_dft:.2e} s = {t_dft/3600:.0f} h = {t_dft/86400:.1f} days; "
      f"MLIP: {t_ml:.0f} s; speed-up = {t_dft/t_ml:.1e}")
# 10x larger cell: DFT ~N^3, MLIP ~N
t_dft10 = nsteps*30.0*1e3; t_ml10 = nsteps*1e-3*10
print(f"  2560 atoms: BOMD ~{t_dft10/86400/365:.0f} years; MLIP {t_ml10:.0f} s; speed-up = {t_dft10/t_ml10:.1e}")
