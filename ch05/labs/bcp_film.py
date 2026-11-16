"""Lab 5.8C - Directed self-assembly (DSA) toy model: A5B5 DPD thin film, Delta a = 15.
Film: 4 L0 x 4 L0 x 3 (periodic; thickness ~ L0/2 so lamellae stand perpendicular), L0 = 6.41
(stress-free period from Lab 5.8B), 5920 beads. Three cases from the same disordered start:
  (i) unguided, (ii) dense pattern  U = -h e cos(2 pi x / L0), h = 0.2,
  (iii) sparse pattern: Gaussian A-attracting stripes (w = 0.8) at pitch 2 L0, h = 1.0  (2x density multiplication).
Order metrics: max S(q)/n and the orientational order P2 = <cos 2 theta>_S over |q| in [0.6, 1.4].
Run: python bcp_film.py  (about 3-4 min)"""
import os, time
import numpy as np
from dpd_core import seed, forces, run_field, make_amat
from bcp_common import build_melt, sq_psi

OUT = os.path.dirname(os.path.abspath(__file__))
L0 = 6.41; N = 10; DA = 15.0

def film(case, t_run=400.0, sd=5):
    box = np.array([4 * L0, 4 * L0, 3.0]); nch = int(round(3 * box.prod() / N))
    rng = np.random.default_rng(sd); seed(sd)
    x, v, typ, bonds = build_melt(nch, N, 0.5, box, rng); e = np.where(typ == 0, 1.0, -1.0)
    A = make_amat(25.0, DA); f = np.zeros_like(x); pt = np.zeros(3)
    forces(x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0, f, pt)
    h, k, mode, w = {"unguided": (0.0, 1.0, 0, 1.0),
                     "dense": (0.2, 2 * np.pi / L0, 0, 1.0),
                     "sparse": (1.0, 2 * np.pi / (2 * L0), 1, 0.8)}[case]
    qs = [(2 * np.pi * a / box[0], 2 * np.pi * b / box[1], 0.0)
          for a in range(-8, 9) for b in range(0, 9) if (b > 0 or a > 0)]
    qv = np.array(qs); qm = np.linalg.norm(qv, axis=1); th = np.arctan2(qv[:, 1], qv[:, 0])
    sel = (qm > 0.6) & (qm < 1.4); hist = []
    nblk = int(t_run / 50)
    for blk in range(nblk):
        run_field(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1250, 10 ** 6, pt, h, k, mode, w)
        s = sq_psi(x, e, qv); j = np.argmax(s)
        P2 = np.sum(s[sel] * np.cos(2 * th[sel])) / np.sum(s[sel])
        hist.append(((blk + 1) * 50, s[j] / len(x), qm[j], P2))
    # composition profile across x (A fraction), for the guided cases
    hA, _ = np.histogram(x[typ == 0, 0], 64, (0, box[0])); hB, _ = np.histogram(x[typ == 1, 0], 64, (0, box[0]))
    return x, typ, box, np.array(hist), hA / np.maximum(hA + hB, 1)

if __name__ == "__main__":
    t0 = time.time(); save = {}
    print(f"film 4L0 x 4L0 x 3, L0={L0}, A5B5, Delta a={DA}, t=400 (10^4 steps, dt=0.04)")
    for case in ("unguided", "dense", "sparse"):
        x, typ, box, hist, prof = film(case)
        t, S, q, P2 = hist[-1]
        print(f"  {case:8s}: t={t:.0f}  max S(q)/n={S:.3f} at |q|={q:.3f} (period {2*np.pi/q:.2f})  P2={P2:+.2f}"
              f"   [t=100: S/n={hist[1,1]:.3f}, P2={hist[1,3]:+.2f}]")
        e = np.where(typ == 0, 1.0, -1.0)
        qq = np.array([[2 * np.pi / (2 * L0), 0, 0], [2 * np.pi / L0, 0, 0], [3 * 2 * np.pi / L0, 0, 0]])
        sx = sq_psi(x, e, qq) / len(x)
        print(f"            S/n along x at q = 2pi/(2L0), 2pi/L0, 3(2pi/L0): {sx[0]:.4f}, {sx[1]:.4f}, {sx[2]:.4f}")
        save[case + "_x"] = x; save[case + "_typ"] = typ; save[case + "_hist"] = hist; save[case + "_prof"] = prof
    save["box"] = box
    np.savez(os.path.join(OUT, "bcp_film.npz"), **save)
    print(f"wall time {time.time()-t0:.0f} s")
