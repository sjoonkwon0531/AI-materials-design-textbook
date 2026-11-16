# Ex 4.7.5: synthetic Delta E_H data -> volcano -> best candidate; then GP-based active search vs random search
import numpy as np, warnings; warnings.filterwarnings("ignore")
kT = 0.0257                                             # eV at 298 K
def activity(dG):  return np.exp(-np.abs(dG)/kT)        # schematic symmetric volcano (relative rate)
# (1) 10 hypothetical metals
rng = np.random.default_rng(42)
names = [f"M{i}" for i in range(10)]; dE = np.round(rng.uniform(-0.8, 0.5, 10), 3); dG = dE + 0.24
order = np.argsort(-activity(dG))
print("(1) 10 hypothetical metals, ranked by volcano activity:")
for i in order: print(f"   {names[i]}: dE_H={dE[i]:+.3f}  dG_H={dG[i]:+.3f}  log10(rel. rate)={np.log10(activity(dG[i])):+.2f}")
# (2) screening a pool of N=500 hypothetical alloys described by 2 cheap descriptors (e.g. d-band centre, coordination)
N = 500
def make_pool(seed):
    g = np.random.default_rng(seed); X = g.uniform(-1, 1, (N, 2))
    dEp = -0.25 + 0.45*X[:,0] - 0.20*X[:,1]**2 + 0.15*np.sin(3*X[:,1]) + 0.03*g.standard_normal(N)  # hidden "DFT"
    return X, dEp + 0.24
def gp_predict(Xt, yt, Xq, ell=0.5, s2=0.1, noise=1e-3):
    """minimal GP regression with fixed RBF hyperparameters (constant mean = sample mean)"""
    k = lambda A,B: s2*np.exp(-0.5*((A[:,None,:]-B[None,:,:])**2).sum(-1)/ell**2)
    m = yt.mean(); K = k(Xt,Xt) + noise*np.eye(len(Xt)); Lc = np.linalg.cholesky(K)
    a = np.linalg.solve(Lc.T, np.linalg.solve(Lc, yt-m)); Ks = k(Xq,Xt)
    v = np.linalg.solve(Lc, Ks.T); return m + Ks@a, np.sqrt(np.maximum(s2 - (v*v).sum(0), 1e-12))
def gp_search(X, dG, n_init=5, budget=100, kappa=1.0, seed=0, tol=None):
    g = np.random.default_rng(seed); best = np.argmin(np.abs(dG))
    idx = list(g.choice(N, n_init, replace=False))
    while len(idx) < budget:
        if tol is None and best in idx: return len(idx)
        if tol is not None and np.min(np.abs(dG[idx])) < tol: return len(idx)
        mu, sd = gp_predict(X[idx], dG[idx], X)
        acq = np.abs(mu) - kappa*sd                       # optimistic (LCB) estimate of |dG_H|
        acq[idx] = np.inf; idx.append(int(np.argmin(acq)))  # label ("DFT") the most promising candidate
    if tol is not None: return len(idx) if np.min(np.abs(dG[idx])) < tol else np.nan
    return len(idx) if best in idx else np.nan
def random_search(dG, seed):
    g = np.random.default_rng(seed); perm = g.permutation(N); return int(np.where(perm == np.argmin(np.abs(dG)))[0][0]) + 1
gp_n, rd_n, gp_t, rd_t, nt = [], [], [], [], []
for s in range(20):
    X, dGp = make_pool(100+s); gp_n.append(gp_search(X, dGp, seed=s)); rd_n.append(random_search(dGp, s))
    gp_t.append(gp_search(X, dGp, seed=s, tol=0.02)); nt.append(np.sum(np.abs(dGp) < 0.02))
    perm = np.random.default_rng(1000+s).permutation(N); rd_t.append(int(np.argmax(np.abs(dGp[perm]) < 0.02)) + 1)
gp_n = np.array(gp_n, float)
print(f"(2) pool N={N}: # 'DFT' calls to find the true best |dG_H| candidate (20 trials)")
print(f"   GP-LCB active search : median {np.nanmedian(gp_n):.0f}, max {np.nanmax(gp_n):.0f}, failures within 100 = {np.sum(np.isnan(gp_n))}")
print(f"   random search        : median {np.median(rd_n):.0f}, max {np.max(rd_n)}   (expected (N+1)/2 = {(N+1)/2:.0f})")
gp_t = np.array(gp_t, float)
print(f"   # calls to find ANY candidate with |dG_H|<0.02 eV (pool contains median {np.median(nt):.0f} such):")
print(f"   GP-LCB: median {np.nanmedian(gp_t):.0f}, max {np.nanmax(gp_t):.0f}; random: median {np.median(rd_t):.0f}, max {np.max(rd_t)}")
