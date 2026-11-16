"""Lab 5.7C - DPD basics (Groot-Warren 1997 conventions, rho=3, a=25, sigma=3, gamma=4.5).
Part 1: temperature control of the modified velocity-Verlet vs dt and lambda.
Part 2: fluctuation-dissipation: break sigma^2 = 2 gamma kT and watch T go to sigma^2/(2 gamma).
Part 3: A/B slab demixing -> Flory-Huggins chi from coexisting compositions -> chi vs Delta a.
Run: python dpd_basics.py   (about 4-5 min)"""
import os, json, math, time
import numpy as np
from dpd_core import seed, forces, run, init_random, make_amat

OUT = os.path.dirname(os.path.abspath(__file__))
NOB = np.zeros((0, 2), np.int64)

def simulate(box, typ, amat, gamma, sigma, dt, lam, t_eq, t_run, every, sd, x=None, v=None):
    box = np.asarray(box, float); n = len(typ)
    rng = np.random.default_rng(sd); seed(sd)
    if x is None:
        x, v = init_random(n, box, rng)
    f = np.zeros_like(x); pt = np.zeros(3)
    forces(x, v, typ, amat, gamma, sigma, dt, box, NOB, 0.0, f, pt)
    run(x, v, f, typ, amat, gamma, sigma, dt, lam, box, NOB, 0.0, int(t_eq / dt), 1000000, pt)
    o = run(x, v, f, typ, amat, gamma, sigma, dt, lam, box, NOB, 0.0, int(t_run / dt), every, pt)
    return x, v, o

def mean_err(a, nb=10):
    a = np.asarray(a); m = len(a) // nb; b = a[: m * nb].reshape(nb, m).mean(1)
    return b.mean(), b.std(ddof=1) / math.sqrt(nb)

if __name__ == "__main__":
    T0 = time.time(); res = {}
    box = (6.0, 6.0, 6.0); n = int(3 * 216); typ = np.zeros(n, np.int64); A = make_amat(25.0, 0.0)
    # ---------------- Part 1
    print("=== Part 1: kinetic temperature vs dt (rho=3, a=25, gamma=4.5, sigma=3, N=648, t=60 after 10) ===")
    res["p1"] = {}
    for lam in (0.5, 0.65):
        row = []
        for dt in (0.01, 0.02, 0.04, 0.06, 0.08):
            _, _, o = simulate(box, typ, A, 4.5, 3.0, dt, lam, 10.0, 60.0, 5, 3)
            kT, ekT = mean_err(o[:, 0]); row.append((dt, kT, ekT))
            P = o[:, 1:].mean()
            if dt == 0.01 and lam == 0.5:
                res["P"] = P
        res["p1"][str(lam)] = row
        print(f"  lambda={lam:4.2f}: " + "  ".join(f"dt={d:.2f}: {k:.4f}" for d, k, _ in row))
    print(f"  pressure at dt=0.01: P = {res['P']:.2f}  (k_BT=1, rho=3, a=25)")
    print(f"  statistical error of each kT ~ {row[0][2]:.4f}")
    # ---------------- Part 2
    print("\n=== Part 2: fluctuation-dissipation check (dt=0.01, lambda=0.5) ===")
    res["p2"] = []
    for gam, sig in ((4.5, 3.0), (4.5, 4.0), (2.25, 3.0), (9.0, 3.0)):
        _, _, o = simulate(box, typ, A, gam, sig, 0.01, 0.5, 10.0, 40.0, 5, 5)
        kT, ekT = mean_err(o[:, 0]); pred = sig * sig / (2 * gam)
        res["p2"].append((gam, sig, kT, ekT, pred))
        print(f"  gamma={gam:4.2f} sigma={sig:3.1f}: kT = {kT:.4f}+/-{ekT:.4f}   sigma^2/(2 gamma) = {pred:.4f}")
    # ---------------- Part 3
    print("\n=== Part 3: A/B slab coexistence, box 16x6x6, N=1728, dt=0.04, lambda=0.65 ===")
    Lx = 16.0; boxs = np.array([Lx, 6.0, 6.0]); n = int(3 * Lx * 36)
    nbin = 64; edges = np.linspace(0, Lx, nbin + 1); xc = 0.5 * (edges[1:] + edges[:-1])
    res["p3"] = []; profiles = {}
    for da in (8.0, 9.0, 10.0, 12.0, 15.0):
        rng = np.random.default_rng(10); seed(10)
        x, v = init_random(n, boxs, rng)
        typ = (x[:, 0] >= Lx / 2).astype(np.int64)            # A left half, B right half
        Am = make_amat(25.0, da); f = np.zeros_like(x); pt = np.zeros(3)
        forces(x, v, typ, Am, 4.5, 3.0, 0.04, boxs, NOB, 0.0, f, pt)
        run(x, v, f, typ, Am, 4.5, 3.0, 0.04, 0.65, boxs, NOB, 0.0, 2500, 1000000, pt)  # t=100 equil.
        hA = np.zeros(nbin); hB = np.zeros(nbin); phis = []
        for blk in range(300):                                  # 300 samples, 50 steps apart (t=600)
            run(x, v, f, typ, Am, 4.5, 3.0, 0.04, 0.65, boxs, NOB, 0.0, 50, 1000000, pt)
            # re-centre on the A slab (circular mean of A positions)
            th = 2 * np.pi * x[typ == 0, 0] / Lx
            cA = (np.arctan2(np.sin(th).mean(), np.cos(th).mean()) * Lx / (2 * np.pi)) % Lx
            xs = (x[:, 0] - cA + Lx / 4) % Lx                   # A slab centred at Lx/4
            a, _ = np.histogram(xs[typ == 0], edges); b, _ = np.histogram(xs[typ == 1], edges)
            hA += a; hB += b
            inA = np.abs(xs - Lx / 4) < 2.0; inB = np.abs(xs - 3 * Lx / 4) < 2.0
            phis.append(0.5 * ((typ[inA] == 1).mean() + (typ[inB] == 0).mean()))
        phi, ephi = mean_err(phis)
        chi = math.log((1 - phi) / phi) / (1 - 2 * phi)
        dchi = abs((math.log((1 - phi - ephi) / (phi + ephi)) / (1 - 2 * (phi + ephi))) - chi)
        res["p3"].append((da, phi, ephi, chi, dchi))
        profiles[da] = (hA / (hA + hB)).tolist()
        print(f"  Delta a={da:5.1f}: minority fraction phi={phi:.4f}+/-{ephi:.4f}  ->  chi={chi:.3f}+/-{dchi:.3f}"
              f"   chi/Delta a={chi/da:.3f}")
    das = np.array([r[0] for r in res["p3"]]); chis = np.array([r[3] for r in res["p3"]])
    slope0 = (das @ chis) / (das @ das)
    p = np.polyfit(das, chis, 1)
    print(f"  fit through origin: chi = {slope0:.3f} Delta a   (Groot-Warren: 0.286 Delta a)")
    print(f"  free linear fit:    chi = {p[0]:.3f} Delta a {p[1]:+.3f}")
    res["slope0"] = slope0; res["fit"] = p.tolist(); res["xc"] = xc.tolist(); res["profiles"] = profiles
    json.dump(res, open(os.path.join(OUT, "dpd_basics.json"), "w"))
    print(f"total wall time {time.time() - T0:.0f} s")
