"""Exercise 9.4.6: widen the pre-training prior to rho ~ U[0, 0.5] and repeat pre-training (4 members),
few-shot LoRA adaptation (K = 4, 16) and the ensemble / residual diagnostics on the reaction family.
Usage: python ex946_prior.py [smoke]"""
import sys, time, importlib, numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, "../../../ch09/labs")
C = importlib.import_module("94_common"); P = importlib.import_module("94b_pretrain")
FS = importlib.import_module("94c_fewshot"); UQ = importlib.import_module("94d_uq")
OUT = "../../../ch09/solutions/b/"
SMOKE = len(sys.argv) > 1
NTR, STEPS, FSTEPS = (64, 60, 60) if SMOKE else (2000, P.STEPS, FS.STEPS)
T_IDX = list(range(4, 41, 4))
t_all = time.process_time()

def make_wide(n_samp, seed, chunk=500):
    """Same draws as make_dataset(kind='pre') (u0, nu, c, beta), then rho ~ U[0, 0.5]."""
    rng = np.random.default_rng(seed)
    u0 = C.sample_ic(n_samp, 256, rng)
    nu, c, beta, rho = C.sample_coeffs(n_samp, rng, "pre")
    rho = rng.uniform(0, 0.5, n_samp)
    us = [C.etdrk4(u0[i:i + chunk], nu[i:i + chunk], c[i:i + chunk], beta[i:i + chunk], rho[i:i + chunk])
          for i in range(0, n_samp, chunk)]
    u = np.concatenate(us)[:, :, ::2]
    return dict(u=u.astype(np.float32), coef=np.stack([nu, c, beta, rho], 1).astype(np.float32))

# (a) dimensionless numbers
print("(a) Peclet |c|/nu and Damkohler rho/nu")
for name, nu, rho in (("pre-training", (0.05, 0.2), (0.0, 0.0)), ("target (reaction)", (0.05, 0.2), (1.0, 3.0)),
                      ("widened prior", (0.05, 0.2), (0.0, 0.5)), ("low-nu set", (0.02, 0.04), (0.0, 0.0))):
    print(f"   {name:18s}: Pe in [0, {1 / nu[0]:.0f}],  Da in [{rho[0] / nu[1]:.1f}, {rho[1] / nu[0]:.1f}]")

# (b) data and pre-training
t0 = time.process_time(); tr = make_wide(NTR, 1); cpu_data = time.process_time() - t0
te_w = make_wide(100, 31)
print(f"(b) widened pre-training set: {NTR} trajectories, rho in [{tr['coef'][:, 3].min():.3f}, {tr['coef'][:, 3].max():.3f}], "
      f"max|u| = {np.abs(tr['u']).max():.2f}, {cpu_data:.1f} CPU-s")
sets = {n: P.load(f) for n, f in (("ID", "pre_test"), ("low-nu", "lownu_test"), ("high-k", "highk_test"), ("reaction", "react_test"))}
members = []
for k in range(P.MEMBERS):
    torch.manual_seed(k)
    m = C.TFNO(P.WIDTH, P.MODES, P.DEPTH)
    t0 = time.process_time(); C.train(m, tr, STEPS, bs=P.BS, lr=P.LR, seed=k); cpu = time.process_time() - t0
    torch.save(m.state_dict(), OUT + f"ex946_member{k}.pt"); members.append(m)
    e = {n: C.traj_error(m, d).mean().item() for n, d in sets.items()}
    print(f"   member {k}: {STEPS} steps, {cpu:.0f} CPU-s | new prior (rho<=0.5) {100 * C.traj_error(m, te_w).mean():.2f}% | "
          + " | ".join(f"{n} {100 * v:.2f}%" for n, v in e.items()) + f" | |lift rho column| = {m.lift.weight[:, 4].norm():.3f}", flush=True)

# (c) few-shot LoRA r = 4 from the new member 0
def pretrained_new():
    m = C.TFNO(P.WIDTH, P.MODES, P.DEPTH); m.load_state_dict(torch.load(OUT + "ex946_member0.pt")); return m
FS.pretrained = pretrained_new
pool, te = P.load("react_train"), sets["reaction"]
val = dict(u=pool["u"][224:], coef=pool["coef"][224:])
err = lambda m, d: 100 * C.traj_error(m, d).mean().item()
print(f"(c) zero-shot on the reaction family, member 0: {err(members[0], te):.2f}%  (Lab 9.13: 46.63%)")
for K in (4, 16):
    runs = {lr: FS.adapt("lora4", FS.subset(pool, K, 0), 0, lr, steps=FSTEPS) for lr in FS.LRS}
    ev = {lr: err(m, val) for lr, (m, _) in runs.items()}; lr = min(ev, key=ev.get)
    errs = [err(runs[lr][0], te)] + [err(FS.adapt("lora4", FS.subset(pool, K, s), s, lr, steps=FSTEPS)[0], te) for s in (1, 2)]
    print(f"   K = {K:2d}: lr selection " + "  ".join(f"{l:.0e}: {v:5.1f}%" for l, v in ev.items()) + f" -> {lr:.0e};  "
          f"LoRA r=4 target-test error {np.mean(errs):.2f} +- {np.std(errs):.2f} %  (3 subsets)", flush=True)

# (d) ensemble spread and residual, old ensemble vs new ensemble
def stats(mem, d):
    ens = UQ.EnsembleMean(mem)
    true = torch.from_numpy(d["u"][:, T_IDX]); preds = torch.stack([C.predict(m, d, T_IDX) for m in mem])
    mu, sd = preds.mean(0), preds.std(0)
    e = (mu - true).reshape(len(true), -1).norm(dim=1) / np.sqrt(true[0].numel())
    spread = sd.reshape(len(true), -1).pow(2).mean(1).sqrt(); res = C.residual_norms(ens, d, T_IDX)
    z = (true - mu).abs() / sd
    return dict(rel=C.rel_l2(mu, true).numpy(), err=e.numpy(), spread=spread.numpy(), res=res.numpy(), sd=sd.mean().item(),
                cov=(z < 1.96).float().mean().item())
old = []
for k in range(P.MEMBERS):
    m = C.TFNO(P.WIDTH, P.MODES, P.DEPTH); m.load_state_dict(torch.load(C.D + f"94_fno_member{k}.pt")); old.append(m.eval())
print("(d) ensemble of 4: old prior (rho = 0) vs widened prior (rho in [0, 0.5])")
print("   set        prior   err(ens)  mean sd   err/sd (median)  cover 1.96sd  rho_S(sd,err)  rms res  err/res (median)")
S = {}
for n, d in sets.items():
    for tag, mem in (("old", old), ("wide", members)):
        s = stats(mem, d); S[(n, tag)] = s
        print(f"   {n:9s}  {tag:5s}  {100 * s['rel'].mean():6.2f}%   {s['sd']:.4f}    {np.median(s['err'] / s['spread']):7.1f}"
              f"         {100 * s['cov']:5.1f}%        {spearmanr(s['spread'], s['err'])[0]:5.2f}        {s['res'].mean():.3f}    {np.median(s['err'] / s['res']):.3f}")
for tag in ("old", "wide"):
    pool_ = {k: np.concatenate([S[(n, tag)][k] for n in sets]) for k in ("rel", "spread", "res")}
    bad = pool_["rel"] > 0.05
    print(f"   pooled 1200, {tag:5s}: {bad.sum()} with rel. error > 5%; AUROC sd {UQ.auroc(pool_['spread'], bad):.3f}, residual {UQ.auroc(pool_['res'], bad):.3f};"
          f" Spearman(sd, err) {spearmanr(pool_['spread'], pool_['rel'])[0]:.2f}")
rho = sets["reaction"]["coef"][:, 3]; s = S[("reaction", "wide")]
for lo, hi in ((1, 2), (2, 3)):
    f = (rho >= lo) & (rho < hi)
    print(f"   reaction set, wide prior, rho in [{lo}, {hi}): rel. error {100 * s['rel'][f].mean():.1f}%, rms spread {s['spread'][f].mean():.4f}, err/sd {np.median((s['err'] / s['spread'])[f]):.1f}")
print(f"   Spearman(rho, spread) on the reaction set: old {spearmanr(rho, S[('reaction', 'old')]['spread'])[0]:.2f}, wide {spearmanr(rho, s['spread'])[0]:.2f}")
print(f"total {time.process_time() - t_all:.0f} CPU-s")
