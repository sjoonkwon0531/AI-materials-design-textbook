"""Lab 6.4.2-6.4.3 — t-SNE and UMAP of the Magpie feature space (matbench_expt_gap,
4,601 compositions), compared with PCA. What is preserved (local neighbours,
labels) and what is not (global distances, cluster sizes), plus seed and
hyperparameter sensitivity, and a synthetic three-cluster test.
Deterministic seeds; ~3-4 min on 2 cores."""
import os, time, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import cross_val_score, KFold
import umap

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
d = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))
feats = [c for c in d.columns if c.startswith("MagpieData")]
X = d[feats].to_numpy(float); y = d.target.to_numpy(float)
keep = X.std(0) > 0
frac_mode = np.array([pd.Series(X[:, j]).value_counts(normalize=True).iloc[0] for j in range(X.shape[1])])
X = X[:, keep & (frac_mode <= 0.99)]
Z = StandardScaler().fit_transform(X); n, p = Z.shape
metal = (y == 0).astype(int)
print(f"n = {n}, p = {p}")

emb = {}
emb["PCA"] = PCA(2).fit_transform(Z)
t = time.time(); emb["t-SNE (perp 30)"] = TSNE(2, perplexity=30, init="pca", random_state=0, n_jobs=2).fit_transform(Z)
print(f"t-SNE time {time.time()-t:.1f} s")
t = time.time(); emb["UMAP (k=15)"] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Z)
print(f"UMAP time {time.time()-t:.1f} s")

rng = np.random.default_rng(0); sub = rng.choice(n, 1000, replace=False)
dh = pdist(Z[sub])
nnh = NearestNeighbors(n_neighbors=11).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
def knn_overlap(E):
    nne = NearestNeighbors(n_neighbors=11).fit(E).kneighbors(E, return_distance=False)[:, 1:]
    return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(nnh, nne)])
cv = KFold(5, shuffle=True, random_state=0)
acc_hd = cross_val_score(KNeighborsClassifier(10), Z, metal, cv=cv).mean()
print(f"\n10-NN metal/non-metal accuracy in the {p}-D standardized space (random 5-fold) = {acc_hd:.3f}")
print(f"{'embedding':18s} {'trust(10)':>10s} {'10-NN overlap':>14s} {'10-NN acc':>10s} {'Spearman(dist)':>15s}")
stats = {}
for k, E in emb.items():
    tw = trustworthiness(Z, E, n_neighbors=10)
    ov = knn_overlap(E)
    acc = cross_val_score(KNeighborsClassifier(10), E, metal, cv=cv).mean()
    rho = spearmanr(dh, pdist(E[sub])).correlation
    stats[k] = (tw, ov, acc, rho)
    print(f"{k:18s} {tw:10.3f} {ov:14.3f} {acc:10.3f} {rho:15.3f}")

print("\n=== perplexity / n_neighbors sweep (t-SNE, UMAP; seed 0) ===")
for perp in [5, 100]:
    E = TSNE(2, perplexity=perp, init="pca", random_state=0, n_jobs=2).fit_transform(Z); emb[f"t-SNE (perp {perp})"] = E
    print(f"t-SNE perp {perp:3d}: trust = {trustworthiness(Z, E, n_neighbors=10):.3f}, "
          f"10-NN overlap = {knn_overlap(E):.3f}, Spearman(dist) = {spearmanr(dh, pdist(E[sub])).correlation:.3f}")
for kk in [5, 50]:
    E = umap.UMAP(n_neighbors=kk, min_dist=0.1, random_state=0).fit_transform(Z); emb[f"UMAP (k={kk})"] = E
    print(f"UMAP k = {kk:3d}     : trust = {trustworthiness(Z, E, n_neighbors=10):.3f}, "
          f"10-NN overlap = {knn_overlap(E):.3f}, Spearman(dist) = {spearmanr(dh, pdist(E[sub])).correlation:.3f}")

print("\n=== seed sensitivity (UMAP k=15) ===")
Es = [emb["UMAP (k=15)"]] + [umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=s).fit_transform(Z) for s in [1, 2]]
for i, E in enumerate(Es[1:], 1):
    _, _, disp = procrustes(Es[0], E)
    print(f"seed 0 vs seed {i}: Procrustes disparity = {disp:.3f}, trust = {trustworthiness(Z, E, n_neighbors=10):.3f}")
emb["UMAP (k=15, seed 1)"] = Es[1]

print("\n=== synthetic test: three Gaussian blobs in 10-D ===")
r = np.random.default_rng(1)
A = r.normal(0, 1.0, (300, 10)); B = r.normal(0, 4.0, (300, 10)); C = r.normal(0, 1.0, (300, 10))
B[:, 0] += 30.0; C[:, 0] += 120.0
S = np.vstack([A, B, C]); lab = np.repeat([0, 1, 2], 300)
def summ(E):
    c = [E[lab == i].mean(0) for i in range(3)]
    rad = [np.sqrt(((E[lab == i] - c[i])**2).sum(1)).mean() for i in range(3)]
    return rad[1] / rad[0], np.linalg.norm(c[2] - c[0]) / np.linalg.norm(c[1] - c[0])
st = {"input (10-D)": summ(S),
      "t-SNE perp 30": summ(TSNE(2, perplexity=30, init="pca", random_state=0, n_jobs=2).fit_transform(S)),
      "UMAP k=15": summ(umap.UMAP(n_neighbors=15, random_state=0).fit_transform(S)),
      "PCA": summ(PCA(2).fit_transform(S))}
for k, (rr, dr) in st.items():
    print(f"{k:14s}: radius(B)/radius(A) = {rr:5.2f}   dist(A,C)/dist(A,B) = {dr:5.2f}")
np.savez(os.path.join(HERE, "64_embed_results.npz"), y=y, **{k.replace(' ', '_'): v for k, v in emb.items()},
         syn_S=S, syn_lab=lab)
print(f"time {time.time()-t0:.1f} s")
