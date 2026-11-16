"""Lab 8.3 (b): parameter-space vs input-space robustness of the band-gap MLPs saved by 83_bandgap_sam.py.
(1) Dinh-style reparameterisation: W1,b1 -> a W1, a b1 ; W2 -> W2 / a leaves a ReLU net's function unchanged
    but rescales Hessian eigenvalues; the scale-normalised lambda_max(T H T) is unchanged.
(2) For every saved model (7 training recipes x 5 seeds): input-gradient norm, MAE under Gaussian input noise,
    MAE under an FGSM (L_inf) attack, and rank correlations with lambda_max and with OOD error.
Usage: python 83_input_robust.py    (~1-2 min)"""
import os, json, time, importlib.util
import numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "83_bandgap_sam.py")).read().split("sub = np.random.default_rng(1)")[0]
src = src.replace('HERE = os.path.dirname(os.path.abspath(__file__))', f'HERE = {HERE!r}')
g = {"__file__": os.path.join(HERE, "83_bandgap_sam.py")}; exec(compile(src, "83_bandgap_sam.py", "exec"), g)
Z, Y, make_mlp, sharpness, S = g["Z"], g["Y"], g["make_mlp"], g["sharpness"], g["S"]
sp = np.load(os.path.join(HERE, "83_bandgap_split.npz")); te, ox, sub = sp["te"], sp["ox"], sp["sub"]
t0 = time.time()


def load(tag, seed):
    m = make_mlp(Z(te[:1]).shape[1]); m.load_state_dict(torch.load(os.path.join(HERE, f"83_bg_{tag}_s{seed}.pt"))); return m


# ---- (0) OOD detection: can we tell oxides from in-distribution test compounds? -------------------------
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
tr_all, ha = sp["tr_all"], sp["ha"]
Ztr = Z(tr_all).numpy(); nnb = NearestNeighbors(n_neighbors=5).fit(Ztr)
knn = lambda idx: nnb.kneighbors(Z(idx).numpy())[0].mean(1)
mu, P = Ztr.mean(0), np.linalg.pinv(np.cov(Ztr.T) + 1e-3 * np.eye(Ztr.shape[1]))
maha = lambda idx: np.sqrt(np.einsum("ij,jk,ik->i", Z(idx).numpy() - mu, P, Z(idx).numpy() - mu))
ens = [load("adamw", s) for s in range(5)]
with torch.no_grad():
    spread = lambda idx: torch.stack([m(Z(idx)).squeeze(-1) for m in ens]).std(0).detach().numpy()
    emean = lambda idx: torch.stack([m(Z(idx)).squeeze(-1) for m in ens]).mean(0).clamp(min=0).detach().numpy()
ybar = Y(tr_all).mean()
print("(0) baseline 'predict the training mean %.3f eV': MAE ID %.3f, oxide %.3f, halide %.3f eV" % (
    ybar, (Y(te) - ybar).abs().mean(), (Y(ox) - ybar).abs().mean(), (Y(ha) - ybar).abs().mean()))
print("(0) OOD detection, ID test (653) vs oxides (1048) / halides (285): AUROC")
for nm, fn in (("5-NN distance", knn), ("Mahalanobis", maha), ("ensemble sd (5 AdamW)", spread)):
    a1 = roc_auc_score(np.r_[np.zeros(len(te)), np.ones(len(ox))], np.r_[fn(te), fn(ox)])
    a2 = roc_auc_score(np.r_[np.zeros(len(te)), np.ones(len(ha))], np.r_[fn(te), fn(ha)])
    print("   %-24s oxide %.3f  halide %.3f" % (nm, a1, a2))
for nm, idx in (("ID test", te), ("oxide", ox)):
    err = np.abs(emean(idx) - Y(idx).numpy()); sd = spread(idx)
    print("   %-8s ensemble MAE %.3f eV, mean sd %.3f eV, coverage of mean +/- 1.96 sd %.1f%%" % (
        nm, err.mean(), sd.mean(), 100 * (err <= 1.96 * sd).mean()))

# ---- (1) reparameterisation -------------------------------------------------------------------------------
m = load("adamw", 0); Xt = Z(te)
with torch.no_grad(): f0 = m(Xt).clone()
print("(1) reparameterisation of AdamW seed-0 model: W1,b1 *= a, W2 /= a")
for a in (1.0, 4.0, 0.25):
    m2 = load("adamw", 0)
    with torch.no_grad():
        m2[0].weight.mul_(a); m2[0].bias.mul_(a); m2[2].weight.div_(a)
        df = (m2(Xt) - f0).abs().max().item()
    print("   a=%-5g max|f'-f| = %.1e eV | lambda_max(H) = %8.2f | lambda_max(THT) = %.4f" % (
        a, df, sharpness(m2, sub), sharpness(m2, sub, adaptive=True)))

# ---- (2) input-space robustness -----------------------------------------------------------------------------
TAGS = [("AdamW (bs 64)", "adamw"), ("AdamW (bs 1024)", "adamw_bs"), ("SAM", "sam"), ("ASAM", "asam"),
        ("F-SAM", "fsam"), ("AdamW + input noise", "adamw_noise"), ("AdamW + FGSM", "adamw_fgsm")]
R = json.load(open(os.path.join(HERE, "83_bandgap_results.json")))["res"]


def metrics(model, idx, seed=0):
    X, y = Z(idx), Y(idx); gen = torch.Generator().manual_seed(seed)
    Xg = X.clone().requires_grad_(True); out = model(Xg).squeeze(-1)
    gx, = torch.autograd.grad(out.sum(), Xg)                              # d f / d x per sample
    r = dict(gnorm=gx.norm(dim=1).mean().item())
    with torch.no_grad():
        base = (model(X).squeeze(-1).clamp(min=0) - y).abs().mean().item()
        for s in (0.1, 0.3):
            r[f"noise{s}"] = np.mean([(model(X + s * torch.randn(X.shape, generator=gen)).squeeze(-1).clamp(min=0) - y)
                                     .abs().mean().item() for _ in range(5)]) - base
    Xa = X.clone().requires_grad_(True)
    la = (model(Xa).squeeze(-1) - y).abs().sum(); ga, = torch.autograd.grad(la, Xa)
    with torch.no_grad():
        for eps in (0.05, 0.1):
            r[f"fgsm{eps}"] = (model(X + eps * ga.sign()).squeeze(-1).clamp(min=0) - y).abs().mean().item() - base
    return r


rows, allr = {}, []
print("(2) input-space robustness on the ID test set (653), increase in MAE [eV]; mean over 5 seeds")
for name, tag in TAGS:
    M = [metrics(load(tag, s), te, seed=s) for s in range(5)]
    A = {k: np.array([x[k] for x in M]) for k in M[0]}
    rows[name] = {k: v.tolist() for k, v in A.items()}
    for s in range(5): allr.append((R[name][s]["lam"], R[name][s]["lam_ad"], A["gnorm"][s], A["fgsm0.1"][s], R[name][s]["id"], R[name][s]["ox"]))
    print("   %-20s |grad_x f| %.3f+/-%.3f | noise 0.1: +%.3f  0.3: +%.3f | FGSM 0.05: +%.3f  0.1: +%.3f | lam_max %.1f" % (
        name, A["gnorm"].mean(), A["gnorm"].std(), A["noise0.1"].mean(), A["noise0.3"].mean(), A["fgsm0.05"].mean(),
        A["fgsm0.1"].mean(), np.mean([R[name][s]["lam"] for s in range(5)])), flush=True)
allr = np.array(allr)
lab = ["lam_max", "lam(THT)", "|grad_x f|", "FGSM0.1 dMAE", "ID MAE", "oxide MAE"]
print("Spearman rank correlations over 35 models:")
for i, j in ((0, 2), (0, 3), (0, 4), (0, 5), (1, 4), (1, 5), (2, 5), (4, 5)):
    rho, p = spearmanr(allr[:, i], allr[:, j]); print("   %-12s vs %-12s rho = %+.2f (p = %.2g)" % (lab[i], lab[j], rho, p))
json.dump(dict(rows=rows, all=allr.tolist(), labels=lab), open(os.path.join(HERE, "83_input_robust_results.json"), "w"))
print("wall %.0f s" % (time.time() - t0))
