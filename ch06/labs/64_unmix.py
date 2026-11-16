"""Lab 6.4.4 — unmixing synthetic XRD-like spectra of three phases with PCA, ICA
(FastICA, plus a from-scratch one-unit FastICA check) and NMF.
All spectra are SYNTHETIC (pseudo-Voigt peaks at invented 2-theta positions).
Two cases: well-separated peaks, and a phase that shares two near-coincident peaks.
Deterministic; ~10 s."""
import os, time, warnings
import numpy as np
from scipy.optimize import linear_sum_assignment, nnls
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA, NMF

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
tt = np.linspace(10, 70, 600)                       # 2-theta grid (deg)

def pv(x, x0, fwhm=0.4, eta=0.5):
    g = np.exp(-4 * np.log(2) * (x - x0)**2 / fwhm**2)
    l = 1 / (1 + 4 * (x - x0)**2 / fwhm**2)
    return eta * l + (1 - eta) * g

def phase(peaks):
    s = sum(h * pv(tt, x0) for x0, h in peaks); return s / s.max()

PEAKS = {
    "separated": [[(20.0, 1.0), (28.5, 0.6), (33.0, 0.8), (47.0, 0.4), (56.0, 0.3)],
                  [(22.0, 0.7), (31.0, 1.0), (38.5, 0.5), (44.5, 0.6), (64.0, 0.3)],
                  [(25.5, 1.0), (35.0, 0.4), (41.0, 0.7), (52.0, 0.5), (60.0, 0.4)]],
    "overlapping": [[(20.0, 1.0), (28.5, 0.6), (33.0, 0.8), (47.0, 0.4), (56.0, 0.3)],
                    [(22.0, 0.7), (31.0, 1.0), (38.5, 0.5), (44.5, 0.6), (64.0, 0.3)],
                    [(25.5, 1.0), (31.2, 0.8), (41.0, 0.7), (44.7, 0.6), (60.0, 0.4)]],
}

def fastica_one_unit(Xw, rng, n_iter=200, tol=1e-10):
    """One-unit FastICA with g = tanh on whitened data Xw (n_obs x k). Returns w (k,)."""
    w = rng.normal(size=Xw.shape[1]); w /= np.linalg.norm(w)
    for it in range(n_iter):
        u = Xw @ w; g = np.tanh(u); gp = 1 - g**2
        w_new = (Xw * g[:, None]).mean(0) - gp.mean() * w
        w_new /= np.linalg.norm(w_new)
        if 1 - abs(w_new @ w) < tol: return w_new, it + 1
        w = w_new
    return w, n_iter

def match(S_true, S_est):
    """Hungarian matching on |corr|; returns per-phase |corr|, sign-fixed estimates in true order."""
    C = np.corrcoef(np.vstack([S_true, S_est]))[:3, 3:]
    r, c = linear_sum_assignment(-np.abs(C))
    est = np.array([np.sign(C[i, j]) * S_est[j] for i, j in zip(r, c)])
    return np.abs(C[r, c]), est

for case, peaks in PEAKS.items():
    rng = np.random.default_rng(0)
    S = np.array([phase(pk) for pk in peaks])                      # 3 x 600 true spectra
    F = rng.dirichlet([1, 1, 1], size=150)                         # 150 x 3 phase fractions (sum to 1)
    X = F @ S + rng.normal(0, 0.01, (150, 600))                    # noisy mixtures
    X_nn = np.clip(X, 0, None)
    print(f"\n=== case: {case} peaks ===")
    print(f"true spectra: pairwise corr = " + ", ".join(f"{np.corrcoef(S)[i, j]:+.3f}" for i, j in [(0, 1), (0, 2), (1, 2)]))
    print(f"excess kurtosis of true spectra (over 2-theta channels): " + ", ".join(f"{kurtosis(s):.1f}" for s in S))
    pca = PCA(3, svd_solver="full").fit(X)
    print(f"rank check: singular values of uncentered X (top 4) = " + ", ".join(f"{v:.2f}" for v in np.linalg.svd(X, compute_uv=False)[:4]))
    print(f"PCA explained variance of first 4 PCs: " + ", ".join(f"{v:.4f}" for v in PCA(4, svd_solver="full").fit(X).explained_variance_ratio_))
    # ICA: observations = 2-theta channels, sources = spectra (spatial-ICA orientation)
    ica = FastICA(3, whiten="unit-variance", random_state=0, max_iter=2000, tol=1e-6)
    ica.fit(X.T)
    S_ica = (X.T @ ica.components_.T).T          # unmixing applied to UNcentered data: keeps the baseline
    nmf = NMF(3, init="nndsvda", random_state=0, max_iter=5000, tol=1e-6)
    W_nmf = nmf.fit_transform(X_nn); S_nmf = nmf.components_       # W 150x3, S 3x600
    res = {}
    for name, Se in [("PCA", pca.components_), ("ICA", S_ica), ("NMF", S_nmf)]:
        cc, Sm = match(S, Se)
        neg = np.array([-s[s < 0].sum() / np.abs(s).sum() for s in Sm])
        # fractions: non-negative least squares of each mixture on the recovered, sign-fixed spectra
        if name == "PCA":
            fmae = np.nan                         # PCs are not spectra: fractions undefined
        else:
            Sn = Sm / np.abs(Sm).max(1, keepdims=True)   # same scale convention as the true spectra (max = 1)
            Fh = np.array([nnls(Sn.T, x)[0] for x in X]); Fh /= Fh.sum(1, keepdims=True); fmae = np.mean(np.abs(Fh - F))
        res[name] = (cc, neg, fmae)
        print(f"{name}: |corr| with truth = " + ", ".join(f"{v:.3f}" for v in cc)
              + f";  negative-intensity share = " + ", ".join(f"{v + 0.0:.2f}" for v in np.maximum(neg, 0.0))
              + (f";  fraction MAE = {res[name][2]:.3f}" if name != "PCA" else ";  fraction MAE = n/a"))
    if case == "separated":
        # from-scratch one-unit FastICA on whitened channels-as-observations
        Xc = X.T - X.T.mean(0); U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        Xw = U[:, :3] * np.sqrt(Xc.shape[0])                     # whitened, unit variance
        w, it = fastica_one_unit(Xw, np.random.default_rng(3))
        u = Xw @ w
        cbest = np.max(np.abs(np.corrcoef(np.vstack([u, S]))[0, 1:]))
        print(f"from-scratch one-unit FastICA (tanh): converged in {it} iterations; "
              f"best |corr| with a true spectrum = {cbest:.3f}; excess kurtosis of u = {kurtosis(u):.1f}")
        for t_ in [0.0, 0.5, 1.0]:
            mix = (1 - t_) * u + t_ * Xw[:, 0]
            print(f"   kurtosis along (1-t)*w + t*PC1 direction, t = {t_}: {kurtosis(mix / mix.std()):.2f}")
        save = dict(tt=tt, S=S, X=X, F=F, S_pca=match(S, pca.components_)[1],
                    S_ica=match(S, S_ica)[1], S_nmf=match(S, S_nmf)[1])
    else:
        save.update(S2=S, X2=X, S2_ica=match(S, S_ica)[1], S2_nmf=match(S, S_nmf)[1], S2_pca=match(S, pca.components_)[1])
np.savez(os.path.join(HERE, "64_unmix_results.npz"), **save)
print(f"time {time.time()-t0:.1f} s")
