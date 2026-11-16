"""Recompute the sharpness numbers of Labs 8.9 and 8.10 from the SAVED models with the converged estimator
(83_sam.lambda_max: scipy eigsh, which='LA', on Hessian-vector products of a float64 copy).

The first version of 83_sam.lambda_max was a power iteration that stopped after 25-40 steps or at a relative
change of 1e-4, which can stop before convergence.  This script
  (1) reloads the 35 band-gap MLPs (83_bg_*.pt), checks that their train/ID/oxide/halide MAEs equal the stored
      ones, recomputes lambda_max(H) and lambda_max(THT) on the same 1,024 training compounds, rewrites
      83_bandgap_results.json and prints the Lab 8.9 summary table in the format of 83_bandgap_sam.py;
  (2) reloads the 9 MLIPs (83_mlip_{adam,sam,asam}_s*.pt), checks their force MAEs, recomputes lambda_max of the
      energy+force loss on the first 64 training frames, rewrites 83_mlip_results.json and prints the Lab 8.10
      per-seed and summary lines in the format of 83_mlip_sam.py.
Nothing is retrained.  Usage: python 83_recompute_sharpness.py   (~5 min on one thread)"""
import os, json, time, importlib.util
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()

# ---- (1) band-gap MLPs (Lab 8.9) ----------------------------------------------------------------------------
src = open(os.path.join(HERE, "83_bandgap_sam.py")).read().split("sub = np.random.default_rng(1)")[0]
src = src.replace('HERE = os.path.dirname(os.path.abspath(__file__))', f'HERE = {HERE!r}')
g = {"__file__": os.path.join(HERE, "83_bandgap_sam.py"), "__name__": "recompute"}
exec(compile(src, "83_bandgap_sam.py", "exec"), g)
Z, Y, make_mlp, sharpness, mae = g["Z"], g["Y"], g["make_mlp"], g["sharpness"], g["mae"]
sp = np.load(os.path.join(HERE, "83_bandgap_split.npz"))
tr_all, te, ox, ha, sub = sp["tr_all"], sp["te"], sp["ox"], sp["ha"], sp["sub"]
RJ = json.load(open(os.path.join(HERE, "83_bandgap_results.json"))); res = RJ["res"]
TAGS = [("AdamW (bs 64)", "adamw"), ("AdamW (bs 1024)", "adamw_bs"), ("SAM", "sam"), ("ASAM", "asam"),
        ("F-SAM", "fsam"), ("AdamW + input noise", "adamw_noise"), ("AdamW + FGSM", "adamw_fgsm")]
print("(1) Lab 8.9 band-gap MLPs: per model lambda_max(H) old -> new | lambda_max(THT) old -> new")
for name, tag in TAGS:
    for s, r in enumerate(res[name]):
        m = make_mlp(Z(te[:1]).shape[1]); m.load_state_dict(torch.load(os.path.join(HERE, f"83_bg_{tag}_s{s}.pt")))
        chk = dict(id=mae(m, te), ox=mae(m, ox), ha=mae(m, ha), train=mae(m, tr_all))
        assert all(abs(chk[k] - r[k]) < 1e-6 for k in chk), (name, s, chk, r)      # same models as the table
        lam, lam_ad = sharpness(m, sub), sharpness(m, sub, adaptive=True)
        r.setdefault("lam_pi", r["lam"]); r.setdefault("lam_ad_pi", r["lam_ad"])   # keep the old power-iteration values
        print("   %-20s seed %d: %7.2f -> %7.2f | %.4f -> %.4f" % (name, s, r["lam_pi"], lam, r["lam_ad_pi"], lam_ad), flush=True)
        r["lam"], r["lam_ad"] = lam, lam_ad
json.dump(RJ, open(os.path.join(HERE, "83_bandgap_results.json"), "w"))
print("Lab 8.9 table (MAE and cpu columns unchanged; sharpness columns recomputed):")
for name, _ in TAGS:
    rows = res[name]; A = {k: np.array([r[k] for r in rows]) for k in ("train", "id", "ox", "ha", "lam", "lam_ad", "cpu")}
    print("%-20s train %.3f | ID %.3f+/-%.3f | oxide %.3f+/-%.3f | halide %.3f+/-%.3f | lam_max %7.1f+/-%5.1f | "
          "lam(THT) %6.2f+/-%5.2f | cpu %3.0f s/model" % (name, A["train"].mean(), A["id"].mean(), A["id"].std(),
          A["ox"].mean(), A["ox"].std(), A["ha"].mean(), A["ha"].std(), A["lam"].mean(), A["lam"].std(),
          A["lam_ad"].mean(), A["lam_ad"].std(), A["cpu"].mean()), flush=True)
print("[%.0f s]" % (time.time() - t0))

# ---- (2) MLIPs (Lab 8.10) ------------------------------------------------------------------------------------
_s = importlib.util.spec_from_file_location("m83", os.path.join(HERE, "83_mlip_sam.py"))
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)
MJ = json.load(open(os.path.join(HERE, "83_mlip_results.json"))); mres, rho = MJ["res"], MJ["rho"]
print("(2) Lab 8.10 MLIPs: per-seed lines with recomputed lambda_max (old value in brackets)")
for name, m, r in (("Adam", "adam", 0.0), ("SAM", "sam", rho), ("ASAM", "asam", 10 * rho)):
    rows = mres[name]
    for seed, row in enumerate(rows):
        mod = M.Model(M.Z, M.Etr.mean().item(), M.Ftr.pow(2).mean().sqrt().item())
        mod.load_state_dict(torch.load(os.path.join(HERE, f"83_mlip_{m}_s{seed}.pt")))
        _, f3 = M.evaluate(mod, M.R3[M.test], M.E3[M.test], M.F3[M.test])
        _, f10 = M.evaluate(mod, M.R10[M.i10], M.E10[M.i10], M.F10[M.i10])
        assert abs(f3 - row["F300"]) < 1e-6 and abs(f10 - row["F1000"]) < 1e-6, (name, seed, f3, f10, row)
        old = row.setdefault("lam_pi", row["lam"]); row["lam"] = M.lam(mod)
        print("   %-5s seed %d: E300 %5.1f meV | F300 %4.0f F600 %4.0f F1000 %4.0f meV/A | lam_max %8.1f | cpu %4.0f s   [%.1f]" % (
            name, seed, 1e3 * row["E300"], 1e3 * row["F300"], 1e3 * row["F600"], 1e3 * row["F1000"], row["lam"], row["cpu"], old),
            flush=True)
    A = {k: np.array([x[k] for x in rows]) for k in ("F300", "F600", "F1000", "lam")}
    print("%-5s (rho %g) F300 %4.0f+/-%2.0f  F600 %4.0f+/-%2.0f  F1000 %4.0f+/-%2.0f meV/A  ratio 1000/300 %.2f  lam_max %.0f+/-%.0f" % (
        name, r, 1e3 * A["F300"].mean(), 1e3 * A["F300"].std(), 1e3 * A["F600"].mean(), 1e3 * A["F600"].std(),
        1e3 * A["F1000"].mean(), 1e3 * A["F1000"].std(), (A["F1000"] / A["F300"]).mean(), A["lam"].mean(), A["lam"].std()), flush=True)
json.dump(MJ, open(os.path.join(HERE, "83_mlip_results.json"), "w"))
print("wall %.0f s" % (time.time() - t0))
