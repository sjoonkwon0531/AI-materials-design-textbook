"""Lab 5.8B - Equilibrium lamellar period D_eq of symmetric A_{N/2}B_{N/2} DPD melts (Delta a = 15)
from the stress-free condition P_N = P_T (normal = lateral pressure).
For each N, two lamellar periods are templated along x in a box (2D) x 6 x 6 (field h=1.0, t=60), the field is
removed (t=60 relax), then <P_xx - (P_yy+P_zz)/2> is averaged over t=320 (8000 steps). Three D values bracket
the root, which is found by a linear fit. Also measures the athermal <Re^2> of each N (statistical segment a).
Run: python bcp_spacing.py   (about 5 min)"""
import os, json, math, time
import numpy as np
from dpd_core import seed, forces, run, run_field, make_amat
from bcp_common import build_melt, sq_psi, chain_stats

OUT = os.path.dirname(os.path.abspath(__file__))
DA = 15.0; LY = 6.0

def dP_at(N, D, sd=2):
    Lx = 2 * D; box = np.array([Lx, LY, LY]); nch = int(round(3 * Lx * LY * LY / N))
    rng = np.random.default_rng(sd); seed(sd)
    x, v, typ, bonds = build_melt(nch, N, 0.5, box, rng)
    A = make_amat(25.0, DA); f = np.zeros_like(x); pt = np.zeros(3)
    forces(x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0, f, pt)
    run_field(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1500, 10 ** 6, pt, 1.0, 2 * np.pi * 2 / Lx)
    run(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1500, 10 ** 6, pt)
    o = run(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 8000, 10, pt)
    dP = o[:, 1] - 0.5 * (o[:, 2] + o[:, 3]); m = len(dP) // 10; b = dP[: m * 10].reshape(10, m).mean(1)
    e = np.where(typ == 0, 1.0, -1.0)
    order = sq_psi(x, e, np.array([[2 * np.pi * 2 / Lx, 0.0, 0.0]]))[0] / len(x)
    return b.mean(), b.std(ddof=1) / math.sqrt(10), order

def athermal_Re2(N, sd=3):
    box = np.array([8.0, 8.0, 8.0]); nch = int(3 * 512 / N)
    rng = np.random.default_rng(sd); seed(sd)
    x, v, typ, bonds = build_melt(nch, N, 0.5, box, rng)
    A = make_amat(25.0, 0.0); f = np.zeros_like(x); pt = np.zeros(3)
    forces(x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0, f, pt)
    run(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1000, 10 ** 6, pt)
    acc = np.zeros(3)
    for _ in range(10):
        run(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 200, 10 ** 6, pt)
        acc += np.array(chain_stats(x, nch, N, box))
    return acc / 10

if __name__ == "__main__":
    t0 = time.time(); rows = []
    print(f"Delta a = {DA}: stress-free lamellar period (box 2D x {LY} x {LY}, rho=3)")
    for N in (8, 10, 14, 20):
        Dg = 6.45 * (N / 10) ** (2 / 3)
        Ds = np.round(Dg * np.array([0.85, 1.0, 1.15]), 2)
        res = [dP_at(N, D) for D in Ds]
        dps = np.array([r[0] for r in res]); errs = np.array([r[1] for r in res])
        p, cov = np.polyfit(Ds, dps, 1, w=1 / errs, cov="unscaled")
        Deq = -p[1] / p[0]
        # error of the root by linear propagation
        g = np.array([p[1] / p[0] ** 2, -1 / p[0]]); dD = math.sqrt(g @ cov @ g)
        Re2, Rg2, b2 = athermal_Re2(N)
        a = math.sqrt(Re2 / (N - 1))
        rows.append(dict(N=N, Ds=Ds.tolist(), dP=dps.tolist(), err=errs.tolist(), Deq=Deq, dD=dD, Re2=Re2, a=a,
                         order=[r[2] for r in res]))
        print(f"  N={N:2d}: " + "  ".join(f"D={D:5.2f}: dP={d:+.3f}+/-{e:.3f}" for D, d, e in zip(Ds, dps, errs))
              + f"  ->  D_eq = {Deq:.2f}+/-{dD:.2f};  athermal <Re^2>={Re2:.2f}, a={a:.3f}, D_eq/Re0={Deq/math.sqrt(Re2):.2f}")
    Ns = np.array([r["N"] for r in rows], float); Deq = np.array([r["Deq"] for r in rows])
    nu = np.polyfit(np.log(Ns), np.log(Deq), 1)[0]
    print(f"power-law fit D_eq ~ N^{nu:.3f}   (strong segregation theory: 2/3; Gaussian coil: 1/2)")
    for chi in (0.286 * DA, DA / 3.27):
        pred = [1.098 * r["a"] * chi ** (1 / 6) * r["N"] ** (2 / 3) for r in rows]
        print(f"  SST D = 1.098 a chi^(1/6) N^(2/3) with chi={chi:.2f}: " + ", ".join(f"{d:.2f}" for d in pred))
    json.dump(dict(rows=rows, nu=nu), open(os.path.join(OUT, "bcp_spacing.json"), "w"))
    print(f"wall time {time.time()-t0:.0f} s")
