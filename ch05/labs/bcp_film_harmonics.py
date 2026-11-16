"""Post-analysis of Lab 5.8C snapshots: Bragg harmonics S(m q0)/n along x, q0 = 2 pi / L0, m = 1..4.
For symmetric (f = 1/2) lamellae the even harmonics vanish (square-wave Fourier series)."""
import os, numpy as np
from bcp_common import sq_psi
d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bcp_film.npz"))
L0 = d["box"][0] / 4
for case in ("dense", "sparse"):
    x = d[case + "_x"]; e = np.where(d[case + "_typ"] == 0, 1.0, -1.0)
    q = np.array([[m * 2 * np.pi / L0, 0, 0] for m in (1, 2, 3, 4)])
    s = sq_psi(x, e, q) / len(x)
    print(f"{case:6s}: S(m q0)/n for m=1..4: " + ", ".join(f"{v:.4f}" for v in s)
          + f"   (ideal sharp square wave: {4/np.pi**2:.4f}, 0, {4/(9*np.pi**2):.4f}, 0)")
