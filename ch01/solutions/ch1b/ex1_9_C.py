"""Ex 1.9.C: histograms of signed magnetization, L=32, T=2.0, 2.27, 3.0 (Metropolis)."""
import numpy as np
from ising_lib import seed, run2d
L = 32; seed(7); rng = np.random.default_rng(7)
yang = lambda T: (1 - np.sinh(2/T)**-4)**(1/8)
for T in [2.0, 2.27, 3.0]:
    spins = rng.choice(np.array([-1, 1], dtype=np.int64), size=(L, L))
    m = run2d(spins, 1/T, 2000, 100000, 1)
    hist, edges = np.histogram(m, bins=40, range=(-1, 1), density=True)
    centers = 0.5*(edges[1:]+edges[:-1])
    flips = np.sum(np.sign(m[1:]) * np.sign(m[:-1]) < 0)
    frac_pos = np.mean(m > 0)
    # peak locations of the |m| histogram and density at m~0 relative to max
    p0 = hist[19:21].mean()/hist.max()
    print(f"T={T}: <m>={m.mean():+.4f}, <|m|>={np.abs(m).mean():.4f}, std(m)={m.std():.4f}, "
          f"frac(m>0)={frac_pos:.3f}, sign changes={flips}, density(m~0)/max={p0:.3f}, "
          f"mode at m={centers[hist.argmax()]:+.3f}, kurtosis U4=1-<m^4>/(3<m^2>^2)={1-np.mean(m**4)/(3*np.mean(m**2)**2):.3f}")
    if T < 2.269: print(f"   Yang M({T}) = {yang(T):.4f}")
    np.savetxt(f"ex1_9_C_hist_T{T}.txt", np.c_[centers, hist], fmt="%.4f")
