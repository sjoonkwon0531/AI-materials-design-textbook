"""Lab 9.4-C: uncertainty of a learned PDE solver.
(1) deep-ensemble spread vs true error, in distribution and under three shifts,
(2) the label-free PDE residual as an error indicator,
(3) split-conformal bands and error bounds,
(4) solver fallback ranked by each indicator,
(5) amortisation break-even with measured CPU times (including a solver at matched accuracy).
Needs 94_data.py and 94b_pretrain.py.  Run: python 94d_uq.py   (about 3 CPU-min)"""
import sys, time, importlib, numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, "../ch09/labs")
C = importlib.import_module("94_common"); P = importlib.import_module("94b_pretrain")
T_IDX = list(range(4, 41, 4))                                # t = 0.1, 0.2, ..., 1.0


class EnsembleMean(nn.Module):
    def __init__(self, members): super().__init__(); self.members = nn.ModuleList(members)
    def forward(self, u0, coef, t): return torch.stack([m(u0, coef, t) for m in self.members]).mean(0)


def auroc(score, label):
    r = np.argsort(np.argsort(score)) + 1.0; n1 = label.sum(); n0 = len(label) - n1
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


if __name__ == "__main__":
    t_all = time.process_time()
    members = []
    for k in range(P.MEMBERS):
        m = C.TFNO(P.WIDTH, P.MODES, P.DEPTH); m.load_state_dict(torch.load(C.D + f"94_fno_member{k}.pt")); members.append(m.eval())
    ens = EnsembleMean(members)
    names = [("cal", "pre_cal"), ("ID", "pre_test"), ("low-nu", "lownu_test"), ("high-k", "highk_test"), ("reaction", "react_test")]
    data = {n: P.load(f) for n, f in names}

    # --- (0) check the residual operator on an exact solution -----------------------------------------
    x = torch.arange(128) * C.L / 128; nu0, c0, k0 = 0.1, 0.7, 3
    class Exact(nn.Module):
        def forward(self, u0, coef, t): return torch.exp(-nu0 * k0 ** 2 * t[:, None]) * torch.sin(k0 * (x[None] - c0 * t[:, None]))
    coef0 = torch.tensor([[nu0, c0, 0.0, 0.0]]); tt = torch.tensor([0.5])
    u, r = C.pde_residual(Exact(), None, coef0, tt)
    print(f"(0) residual operator on the exact solution exp(-nu k^2 t) sin k(x - ct): rms r = {r.pow(2).mean().sqrt():.1e} "
          f"(rms u_t = {(nu0 * k0 ** 2 + c0 * k0) * u.pow(2).mean().sqrt():.2f})")

    # --- (1)+(2) per-trajectory error, spread, residual ----------------------------------------------
    S = {}
    print("\n(1)(2) ensemble of 4: error, spread, residual (300 trajectories each, block t = 0.1..1.0)")
    print("  set        err(member0)  err(ens)   mean sd    cover 1.96sd  cover 3.18sd   rho_S(sd,err)  rho_S(res,err)   rms res")
    for n, d in data.items():
        true = torch.from_numpy(d["u"][:, T_IDX])
        preds = torch.stack([C.predict(m, d, T_IDX) for m in members])           # (M, N, 10, 128)
        mu, sd = preds.mean(0), preds.std(0)
        err = (mu - true).reshape(len(true), -1).norm(dim=1) / np.sqrt(true[0].numel())   # RMS error per trajectory
        rel = C.rel_l2(mu, true); rel0 = C.rel_l2(preds[0], true)
        spread = sd.reshape(len(true), -1).pow(2).mean(1).sqrt()                  # RMS ensemble sd per trajectory
        res = C.residual_norms(ens, d, T_IDX)                                     # RMS PDE residual (no labels)
        z = (true - mu).abs() / sd
        S[n] = dict(err=err.numpy(), rel=rel.numpy(), spread=spread.numpy(), res=res.numpy(),
                    zmax=z.reshape(len(true), -1).max(1).values.numpy(), cov196=(z < 1.96).float().mean().item(),
                    emax=(true - mu).abs().reshape(len(true), -1).max(1).values.numpy(),
                    rms_u=true.pow(2).mean().sqrt().item(), sd_mean=sd.mean().item())
        if n == "ID":
            np.savez(C.D + "94d_example.npz", true=true[:8].numpy(), mu=mu[:8].numpy(), sd=sd[:8].numpy())
        print(f"  {n:9s}   {100 * rel0.mean():6.2f}%    {100 * rel.mean():6.2f}%   {sd.mean():.4f}     {100 * (z < 1.96).float().mean():5.1f}%"
              f"        {100 * (z < 3.18).float().mean():5.1f}%         {spearmanr(spread, err)[0]:5.2f}          {spearmanr(res, err)[0]:5.2f}"
              f"        {res.mean():.3f}")
    test = ["ID", "low-nu", "high-k", "reaction"]
    pool = {k: np.concatenate([S[n][k] for n in test]) for k in ("err", "rel", "spread", "res")}
    print(f"  pooled (1200): Spearman(sd, err) = {spearmanr(pool['spread'], pool['err'])[0]:.2f}, "
          f"Spearman(res, err) = {spearmanr(pool['res'], pool['err'])[0]:.2f}; "
          f"Pearson of logs: sd {pearsonr(np.log(pool['spread']), np.log(pool['err']))[0]:.2f}, "
          f"res {pearsonr(np.log(pool['res']), np.log(pool['err']))[0]:.2f}")
    bad = pool["rel"] > 0.05
    print(f"  detecting 'relative error > 5%' ({bad.sum()} of 1200): AUROC sd {auroc(pool['spread'], bad):.3f}, "
          f"residual {auroc(pool['res'], bad):.3f}")
    for n in test:
        print(f"    {n:9s} error/spread = {np.median(S[n]['err'] / S[n]['spread']):5.1f},  error/residual = "
              f"{np.median(S[n]['err'] / S[n]['res']):.3f}  (medians)")

    # --- (3) split conformal, calibrated on the ID calibration set (n = 300), alpha = 0.1 --------------
    def qhat(s, alpha=0.1):
        k = int(np.ceil((len(s) + 1) * (1 - alpha))); return np.sort(s)[k - 1]
    q_loc = qhat(S["cal"]["zmax"]); q_glob = qhat(S["cal"]["emax"] / S["cal"]["spread"])
    q_sd = qhat(S["cal"]["err"] / S["cal"]["spread"]); q_res = qhat(S["cal"]["err"] / S["cal"]["res"])
    print(f"\n(3) split conformal (calibration: 300 ID trajectories, nominal 90%)")
    print(f"  whole-field bands |u - mu| <= q * scale for all (x, t):  local scale sd(x,t): q = {q_loc:.1f};  "
          f"trajectory scale rms(sd): q = {q_glob:.2f}")
    print(f"  error-norm bounds:  ||e|| <= q ||sd||: q = {q_sd:.2f};  ||e|| <= q ||r||: q = {q_res:.3f}")
    print("  set        band, local scale     band, trajectory scale   bound from spread   bound from residual")
    print("             (coverage; half-width / rms u)                  (coverage; bound / error, median)")
    for n in test:
        s = S[n]
        print(f"  {n:9s}   {100 * (s['zmax'] <= q_loc).mean():5.1f}% ({q_loc * s['sd_mean'] / s['rms_u']:.3f})"
              f"         {100 * (s['emax'] <= q_glob * s['spread']).mean():5.1f}% ({q_glob * s['spread'].mean() / s['rms_u']:.3f})"
              f"          {100 * (s['err'] <= q_sd * s['spread']).mean():5.1f}% ({np.median(q_sd * s['spread'] / s['err']):4.1f}x)"
              f"       {100 * (s['err'] <= q_res * s['res']).mean():5.1f}% ({np.median(q_res * s['res'] / s['err']):4.1f}x)")

    # --- (4) hybrid: send the most suspicious trajectories to the solver --------------------------------
    print("\n(4) solver fallback on the pooled 1200 trajectories (mean relative error of what remains with the surrogate "
          "counted as 0 for re-solved cases)")
    rng = np.random.default_rng(0)
    print("  fallback fraction     random    ensemble sd   PDE residual   oracle")
    for f in (0.0, 0.1, 0.25, 0.5):
        nfb = int(f * 1200); row = []
        for key in ("random", "spread", "res", "rel"):
            sc = rng.random(1200) if key == "random" else pool[key]
            keep = np.argsort(sc)[:1200 - nfb]
            row.append(100 * pool["rel"][keep].sum() / 1200)
        print(f"        {f:4.2f}           {row[0]:5.2f}%      {row[1]:5.2f}%       {row[2]:5.2f}%      {row[3]:5.2f}%")

    # --- (5) amortisation break-even with measured CPU times -------------------------------------------
    d = data["ID"]; u0 = torch.from_numpy(d["u"][:, 0]); coef = torch.from_numpy(d["coef"]); t1 = torch.ones(300)
    def cpu_ms(fn, rep=5):
        t0 = time.process_time(); [fn() for _ in range(rep)]; return 1e3 * (time.process_time() - t0) / rep / 300
    with torch.no_grad():
        c_inf = cpu_ms(lambda: members[0](u0, coef, t1)); c_ens = cpu_ms(lambda: ens(u0, coef, t1))
    c_res = cpu_ms(lambda: C.pde_residual(members[0], u0, coef, t1))
    # solvers: regenerate the ID test initial conditions (seed 3) and solve at several (n, dt)
    rng = np.random.default_rng(3); u0_256 = C.sample_ic(300, 256, rng); cf = C.sample_coeffs(300, rng, "pre")
    ref = d["u"][:, -1]                                                            # reference u(T) on 128 points
    print("\n(5) cost per query u0 -> u(T), CPU-ms per trajectory (batch of 300, one thread)")
    print(f"  FNO member 0: {c_inf:.3f} | ensemble of 4: {c_ens:.3f} | member 0 + residual (forward-mode AD): {c_res:.3f}")
    fno_err = np.sqrt(((C.predict(members[0], d, [40])[:, 0].numpy() - ref) ** 2).sum(1) / (ref ** 2).sum(1)).mean()
    print(f"  FNO member 0 relative error at T: {100 * fno_err:.2f}%")
    sol = {}
    for n, dt in [(256, 5e-3), (64, 2e-2), (32, 5e-2), (32, 1e-1), (16, 1e-1)]:
        u0n = u0_256[:, ::256 // n]
        t0 = time.process_time(); uT = [C.etdrk4(u0n, *cf, dt=dt, n_save=1)[:, -1] for _ in range(3)][0]
        ms = 1e3 * (time.process_time() - t0) / 3 / 300
        uT128 = np.fft.irfft(np.fft.rfft(uT) * (128 / n), n=128)                   # spectral interpolation to 128 points
        e = np.sqrt(((uT128 - ref) ** 2).sum(1) / (ref ** 2).sum(1)).mean()
        sol[(n, dt)] = (ms, e)
        print(f"  ETDRK4 n = {n:3d}, dt = {dt:5.3f} ({int(round(1 / dt)):3d} steps): {ms:7.3f} ms, relative error at T {100 * e:.2e}%")
    c_ref = float(np.load(C.D + "94_solver_ms.npy")[0]); c_tr = float(np.load(C.D + "94_train_cpu.npy").mean())
    c_data = 2000 * c_ref / 1e3
    print(f"  training set: 2000 reference solves = {c_data:.0f} CPU-s; training one member = {c_tr:.0f} CPU-s")
    for label, cs in [("reference solver (256, 5e-3)", sol[(256, 5e-3)][0])] + \
                     [(f"solver at matched accuracy (n={n}, dt={dt})", v[0]) for (n, dt), v in sol.items() if v[1] <= fno_err][-1:]:
        for who, ci, ct in [("single FNO", c_inf, c_data + c_tr), ("ensemble of 4", c_ens, c_data + 4 * c_tr)]:
            nstar = 1e3 * ct / (cs - ci) if cs > ci else float("inf")
            print(f"  break-even vs {label:42s} {who:13s}: speed-up {cs / ci:6.1f}x, N* = {nstar:,.0f} queries")
    print(f"total {time.process_time() - t_all:.0f} CPU-s")
    np.savez(C.D + "94d_uq.npz", **{f"{n}|{k}": v for n, s in S.items() for k, v in s.items()},
             q=np.array([q_loc, q_glob, q_sd, q_res]), cost=np.array([c_inf, c_ens, c_res, c_data, c_tr, fno_err]),
             sol=np.array([[n, dt, ms, e] for (n, dt), (ms, e) in sol.items()]))
