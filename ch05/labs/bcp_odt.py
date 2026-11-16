"""Lab 5.8A (part 2) - Bracketing the order-disorder transition (ODT) of DPD A5B5 by hysteresis.
Box 12 x 6 x 6 (1296 beads). For each Delta a, two runs of t=400 (10^4 steps) field-free:
  'heat'  : start from a templated lamellar state (2 periods of 6.0 along x, field h=0.5 for t=40),
  'cool'  : start from a disordered random-walk melt.
Order parameter: Psi = max_q S(q)/n, averaged over the second half of each run.
Run: python bcp_odt.py   (about 3-4 min)"""
import os, json, time
import numpy as np
from dpd_core import seed, forces, run_field, make_amat
from bcp_common import build_melt, sq_psi

OUT = os.path.dirname(os.path.abspath(__file__))
N = 10; box = np.array([12.0, 6.0, 6.0])

def qgrid(box, nmax=4):
    return np.array([(2 * np.pi * a / box[0], 2 * np.pi * b / box[1], 2 * np.pi * c / box[2])
                     for a in range(-2 * nmax, 2 * nmax + 1) for b in range(-nmax, nmax + 1)
                     for c in range(-nmax, nmax + 1)
                     if (a, b, c) > (0, 0, 0) and 0.5 < np.sqrt((2*np.pi*a/box[0])**2 + (2*np.pi*b/box[1])**2
                                                                + (2*np.pi*c/box[2])**2) < 2.2])

def one(da, start, sd=31, t_run=400.0):
    nch = int(round(3 * box.prod() / N)); rng = np.random.default_rng(sd); seed(sd)
    x, v, typ, bonds = build_melt(nch, N, 0.5, box, rng); e = np.where(typ == 0, 1.0, -1.0)
    A = make_amat(25.0, da); f = np.zeros_like(x); pt = np.zeros(3)
    forces(x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0, f, pt)
    if start == "heat":
        run_field(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1000, 10 ** 6, pt,
                  0.5, 2 * np.pi * 2 / box[0])
    qv = qgrid(box); psi = []
    nblk = int(t_run / 8.0)
    for blk in range(nblk):
        run_field(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 200, 10 ** 6, pt, 0.0, 1.0)
        if blk >= nblk // 2:
            psi.append(sq_psi(x, e, qv).max() / len(x))
    return float(np.mean(psi)), float(np.std(psi))

if __name__ == "__main__":
    t0 = time.time(); rows = []
    print(f"box {box.tolist()}, A5B5, {int(round(3*box.prod()))} beads; Psi = max S(q)/n (2nd half of t=400)")
    for da in (6.0, 7.0, 8.0, 9.0, 10.0, 12.0):
        h = one(da, "heat"); c = one(da, "cool")
        rows.append((da, h[0], h[1], c[0], c[1]))
        print(f"  Delta a={da:4.1f} (chi_GW N={2.86*da:5.1f}): from lamellae Psi={h[0]:.4f}+/-{h[1]:.4f}   "
              f"from disorder Psi={c[0]:.4f}+/-{c[1]:.4f}")
    json.dump(rows, open(os.path.join(OUT, "bcp_odt.json"), "w"))
    print(f"wall time {time.time()-t0:.0f} s")
