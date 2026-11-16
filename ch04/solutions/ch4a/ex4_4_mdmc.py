"""Verification code for Ch.4 exercise 4.4.5 (MD vs Metropolis MC on the Lab 4.4 LJ system) and the
MD diffusion coefficient. Run time: about 10-15 min on 2 cores."""
import time
import numpy as np
import sympy as sp

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


def _unused_nve_run(nsteps, dt, seed=1, nside=6, density=0.7, T0=1.0, every=5):
    """Lab 4.4 Part B: identical initial condition (seed 1 is the first random draw in lab_4_4.py)."""
    rng = np.random.RandomState(seed)
    pos, L, N = init_lattice(nside, density)
    vel = rng.randn(N, 2); vel -= vel.mean(axis=0)
    _, T = kinetic_temp(vel); vel *= np.sqrt(T0/T)
    F, PE = forces_energy(pos, L); Es = []
    for s in range(nsteps):
        vel += 0.5*dt*F; pos = (pos + dt*vel) % L; F, PE = forces_energy(pos, L); vel += 0.5*dt*F
        if s % every == 0:
            Es.append(0.5*np.sum(vel*vel) + PE)
    return np.array(Es), pos, vel, L


# =========================== 4.4.5 MD vs MC ===========================
print("\n=== 4.4.5 MD (NVT) vs Metropolis MC, 2D LJ N=36 rho=0.7 T=1.0 ===")
T0 = 1.0; beta = 1.0/T0
nbins = 50; rmax = None


def gr_accum(pos, L, hist, edges):
    rij = pos[:, None, :] - pos[None, :, :]; rij -= L*np.round(rij/L)
    r = np.sqrt(np.sum(rij*rij, axis=2))[np.triu_indices(len(pos), 1)]
    hist += np.histogram(r, bins=edges)[0]


def gr_normalise(hist, nconf, N, L, edges):
    rho = N/L**2; shell = np.pi*(edges[1:]**2 - edges[:-1]**2)
    return hist/(nconf*0.5*N*rho*shell)


def block_err(a, nb=10):
    a = np.asarray(a); m = len(a)//nb
    b = a[:m*nb].reshape(nb, m).mean(axis=1)
    return b.mean(), b.std(ddof=1)/np.sqrt(nb)


pos0, L, N = init_lattice(6, 0.7)
edges = np.linspace(0.0, L/2, nbins + 1); rc_mid = 0.5*(edges[1:] + edges[:-1])


def md_nvt(thermostat, nequil, nprod, dt=0.004, seed=2, gamma=1.0):
    rng = np.random.RandomState(seed)
    pos = pos0.copy(); vel = rng.randn(N, 2); vel -= vel.mean(axis=0)
    vel *= np.sqrt(T0/kinetic_temp(vel)[1])
    F, PE = forces_energy(pos, L); U = []; hist = np.zeros(nbins); nconf = 0
    c1 = np.exp(-gamma*dt); c2 = np.sqrt((1 - c1*c1)*T0)
    for s in range(nequil + nprod):
        if thermostat == "rescale":            # Lab 4.4 Part C
            vel += 0.5*dt*F; pos = (pos + dt*vel) % L; F, PE = forces_energy(pos, L); vel += 0.5*dt*F
            if s % 20 == 0:
                vel *= np.sqrt(T0/kinetic_temp(vel)[1])
        else:                                   # Langevin BAOAB (exact canonical up to O(dt^2))
            vel += 0.5*dt*F; pos = (pos + 0.5*dt*vel) % L
            vel = c1*vel + c2*rng.randn(N, 2)
            pos = (pos + 0.5*dt*vel) % L; F, PE = forces_energy(pos, L); vel += 0.5*dt*F
        if s >= nequil and s % 10 == 0:
            U.append(PE/N)
            if s % 50 == 0:
                gr_accum(pos, L, hist, edges); nconf += 1
    return np.array(U), gr_normalise(hist, nconf, N, L, edges)


def particle_energy(pos, i, ri, L):
    d = pos - ri; d -= L*np.round(d/L)
    r2 = np.sum(d*d, axis=1); r2[i] = np.inf
    m = r2 < rc2; inv6 = 1.0/r2[m]**3
    return np.sum(4.0*(inv6*inv6 - inv6) - ushift)


def mc_nvt(nequil_sweeps, nprod_sweeps, delta=0.15, seed=3):
    rng = np.random.RandomState(seed)
    pos = pos0.copy(); _, PE = forces_energy(pos, L)
    U = []; hist = np.zeros(nbins); nconf = 0; acc = 0; tried = 0
    for sweep in range(nequil_sweeps + nprod_sweeps):
        for _ in range(N):
            i = rng.randint(N)
            new = (pos[i] + delta*(2*rng.rand(2) - 1)) % L
            dE = particle_energy(pos, i, new, L) - particle_energy(pos, i, pos[i], L)
            tried += 1
            if dE < 0 or rng.rand() < np.exp(-beta*dE):     # Metropolis: rand < exp(-beta dE)
                pos[i] = new; PE += dE; acc += 1
        if sweep >= nequil_sweeps:
            U.append(PE/N)
            if sweep % 5 == 0:
                gr_accum(pos, L, hist, edges); nconf += 1
    _, PEchk = forces_energy(pos, L)
    return np.array(U), gr_normalise(hist, nconf, N, L, edges), acc/tried, abs(PEchk - PE)


t0 = time.time()
U_mc, g_mc, accr, ebook = mc_nvt(2000, 40000)
print(f"  MC: acceptance={accr:.2f}, running-energy bookkeeping error={ebook:.1e}, time {time.time()-t0:.0f}s")
t0 = time.time()
U_lg, g_lg = md_nvt("langevin", 5000, 200000)
print(f"  MD Langevin done, time {time.time()-t0:.0f}s")
t0 = time.time()
U_rs, g_rs = md_nvt("rescale", 5000, 200000)
print(f"  MD rescale done, time {time.time()-t0:.0f}s")
for name, Uarr in [("MC Metropolis", U_mc), ("MD Langevin (BAOAB)", U_lg), ("MD vel. rescale (Lab C)", U_rs)]:
    m, e = block_err(Uarr)
    print(f"  {name:24s} <U>/N = {m:.4f} +/- {e:.4f}")
ipk = np.argmax(g_mc)
print(f"  g(r) first peak: MC r={rc_mid[ipk]:.3f} g={g_mc[ipk]:.3f} | Langevin g={g_lg[ipk]:.3f} | rescale g={g_rs[ipk]:.3f}")
sel = rc_mid > 0.8
print(f"  max|g_MD(Langevin)-g_MC| for r>0.8: {np.max(np.abs(g_lg[sel]-g_mc[sel])):.3f};  "
      f"max|g_MD(rescale)-g_MC|: {np.max(np.abs(g_rs[sel]-g_mc[sel])):.3f}")
np.savez("ex4_4_gr.npz", r=rc_mid, g_mc=g_mc, g_lg=g_lg, g_rs=g_rs)

# diffusion coefficient: NVE after Langevin equilibration, unwrapped MSD
print("\n  Diffusion from MD (NVE after NVT equilibration, unwrapped coordinates):")
rng = np.random.RandomState(4)
pos = pos0.copy(); vel = rng.randn(N, 2); vel -= vel.mean(axis=0); vel *= np.sqrt(T0/kinetic_temp(vel)[1])
F, _ = forces_energy(pos, L); dtd = 0.004
c1 = np.exp(-dtd); c2 = np.sqrt((1 - c1*c1)*T0)
for s in range(20000):
    vel += 0.5*dtd*F; pos = (pos + 0.5*dtd*vel) % L; vel = c1*vel + c2*rng.randn(N, 2)
    pos = (pos + 0.5*dtd*vel) % L; F, _ = forces_energy(pos, L); vel += 0.5*dtd*F
vel -= vel.mean(axis=0); vel *= np.sqrt(T0/kinetic_temp(vel)[1])   # start NVE exactly at T0
unw = pos.copy(); start = unw.copy(); msd_t = []; Ts = []
for s in range(1, 25001):
    vel += 0.5*dtd*F; step = dtd*vel; unw += step; pos = (pos + step) % L
    F, _ = forces_energy(pos, L); vel += 0.5*dtd*F
    if s % 250 == 0:
        msd_t.append((s*dtd, np.mean(np.sum((unw - start)**2, axis=1)))); Ts.append(kinetic_temp(vel)[1])
msd_t = np.array(msd_t)
fit = msd_t[msd_t[:, 0] >= 20.0]
slope = np.polyfit(fit[:, 0], fit[:, 1], 1)[0]
print(f"  NVE mean T={np.mean(Ts):.3f}; MSD(t=10)={msd_t[msd_t[:,0]==10.0][0,1]:.3f}, MSD(t=100)={msd_t[-1,1]:.3f}; "
      f"D = slope/4 (t in [20,100]) = {slope/4:.4f} (LJ units)")
