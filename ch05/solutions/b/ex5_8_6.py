"""Ex 5.8.6(b): dense chemical pattern of period P = s*L0 on an A5B5 DPD film, box 4P x 4L0 x 3.
Same protocol as Lab 5.15 (Delta a=15, h=0.2, t=400, seed 5). Prints max S/n, its q-vector, P2,
S/n along x at the pattern wave number. Usage: python ex5_8_6.py 1.0 1.1 1.2:0.05 ...  (period factor[:field h[:seed]])"""
import sys, os, time
import numpy as np
sys.path.insert(0, "../../../ch05/labs")
from dpd_core import seed, forces, run_field, make_amat
from bcp_common import build_melt, sq_psi
L0 = 6.41; N = 10; DA = 15.0

def film(s, t_run=400.0, sd=5, h=0.2):
    P = s * L0
    box = np.array([4 * P, 4 * L0, 3.0]); nch = int(round(3 * box.prod() / N))
    rng = np.random.default_rng(sd); seed(sd)
    x, v, typ, bonds = build_melt(nch, N, 0.5, box, rng); e = np.where(typ == 0, 1.0, -1.0)
    A = make_amat(25.0, DA); f = np.zeros_like(x); pt = np.zeros(3)
    forces(x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0, f, pt)
    k = 2 * np.pi / P
    idx = [(a, b) for a in range(-12, 13) for b in range(0, 9) if (b > 0 or a > 0)]
    qv = np.array([(2 * np.pi * a / box[0], 2 * np.pi * b / box[1], 0.0) for a, b in idx])
    qm = np.linalg.norm(qv, axis=1); th = np.arctan2(qv[:, 1], qv[:, 0])
    sel = (qm > 0.6) & (qm < 1.4); selb = (qm > 0.45) & (qm < 1.9); hist = []
    for blk in range(int(t_run / 50)):
        run_field(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1250, 10 ** 6, pt, h, k, 0, 1.0)
        sq = sq_psi(x, e, qv); j = np.argmax(sq)
        P2 = np.sum(sq[sel] * np.cos(2 * th[sel])) / np.sum(sq[sel])
        P2b = np.sum(sq[selb] * np.cos(2 * th[selb])) / np.sum(sq[selb])
        hist.append(((blk + 1) * 50, sq[j] / len(x), qm[j], P2, idx[j], P2b, sq / len(x)))
    sx = sq_psi(x, e, np.array([[k, 0, 0], [3 * k, 0, 0]])) / len(x)
    return hist, sx, len(x), idx

if __name__ == "__main__":
    for arg in (sys.argv[1:] or ["1.1", "1.2", "1.3"]):   # "s[:h[:seed]]"; default = the exercise's set
        p = arg.split(":"); s = float(p[0]); hh = float(p[1]) if len(p) > 1 else 0.2; sd = int(p[2]) if len(p) > 2 else 5
        t0 = time.time()
        hist, sx, n, idx = film(s, h=hh, sd=sd)
        t, S, q, P2, nn, P2b, sq = hist[-1]
        top = np.argsort(sq)[::-1][:3]
        print(f"seed {sd} h={hh:.2f} pattern period {s:.2f} L0 (box {4*s*L0:.2f} x {4*L0:.2f} x 3, {n} beads): t=400 max S/n={S:.3f} "
              f"at n=({nn[0]},{nn[1]}) |q|={q:.3f} (period {2*np.pi/q:.2f} = {2*np.pi/q/L0:.3f} L0)  P2(0.6-1.4)={P2:+.2f}; "
              f"S/n at q_pattern={sx[0]:.3f}, at 3q_pattern={sx[1]:.4f}; [t=100: S/n={hist[1][1]:.3f} P2={hist[1][3]:+.2f}; "
              f"t=200: S/n={hist[3][1]:.3f} P2={hist[3][3]:+.2f}]; P2(|q| in 0.45-1.9)={P2b:+.2f}; top peaks "
              + "; ".join(f"n={idx[i]} S/n={sq[i]:.3f} period/L0={2*np.pi/np.hypot(2*np.pi*idx[i][0]/(4*s*L0), 2*np.pi*idx[i][1]/(4*L0))/L0:.3f}" for i in top)
              + f"  ({time.time()-t0:.0f} s)", flush=True)
