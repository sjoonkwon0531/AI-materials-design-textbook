"""Ch.7.7 case study, step 1: train DirectMLP (with/without rotation augmentation), SchNetLike and the
NequIP-style e3nn model on N = 100, 300, 1000 ethanol GFN2-xTB frames at 300 K (equal step budgets),
then evaluate on held-out 300 K frames, on 600 K / 1000 K frames (extrapolation), on randomly rotated
300 K frames, and measure the rotation-equivariance error. Saves weights 77_<kind>_N<n>.pt and 77_results.json.
Usage: python 77_train.py [N ...]   (all three sizes: ~8-10 CPU-min on one thread; wall time grows under load)."""
import os, sys, json, time, importlib.util
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("m77", os.path.join(HERE, "77_models.py"))
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)

R3, E3, F3, Z = M.load(300); R6, E6, F6, _ = M.load(600); R10, E10, F10, _ = M.load(1000)
perm = np.random.default_rng(0).permutation(2000)              # training pool: frames 0-1999 (first 10 ps)
val = slice(2000, 2100)                                         # validation: frames 2000-2099
test = slice(2200, 3000)                                        # test: frames 2200-2999 (2 ps gap after pool)
i6 = np.sort(np.random.default_rng(1).choice(3000, 500, replace=False))
i10 = np.sort(np.random.default_rng(2).choice(1000, 500, replace=False))
STEPS = {100: 800, 300: 1200, 1000: 2000}                       # GNNs: batch 8, equal budget at given N
MLP_STEPS, MLP_BS, MLP_LR = 4000, 32, 1e-3                      # MLPs are ~30x cheaper: give them 4000 x 32
LR = {"schnet": 1e-2, "nequip": 1e-2}                    # chosen on validation by 77_lr_select.py
KINDS = [("mlp", False), ("mlp_aug", True), ("mlp_frame", False), ("schnet", False), ("nequip", False)]
sizes = [int(a) for a in sys.argv[1:]] or [100, 300, 1000]
path = os.path.join(HERE, "77_results.json")
res = json.load(open(path)) if os.path.exists(path) else {}
print("E std 300/600/1000 K: %.3f %.3f %.3f eV; |F| mean %.3f %.3f %.3f eV/A" % (
    E3.std(), E6.std(), E10.std(), F3.abs().mean(), F6.abs().mean(), F10.abs().mean()))
for N in sizes:
    tr = torch.as_tensor(perm[:N])
    for kind, aug in KINDS:
        t0 = time.time(); c0 = time.process_time(); log = []
        mlp = kind.startswith("mlp")
        model = M.train("mlp" if kind == "mlp_aug" else kind, R3[tr], E3[tr], F3[tr], R3[val], E3[val], F3[val], Z,
                        steps=MLP_STEPS if mlp else STEPS[N], bs=MLP_BS if mlp else 8,
                        lr=MLP_LR if mlp else LR[kind], augment=aug, seed=0, log=log)
        cpu = time.process_time() - c0
        torch.save(model.state_dict(), os.path.join(HERE, f"77_{kind}_N{N}.pt"))
        r = {"params": sum(p.numel() for p in model.parameters()), "cpu_s": cpu, "wall_s": time.time() - t0, "log": log}
        for tag, (R, E, F) in {"300K": (R3[test], E3[test], F3[test]), "600K": (R6[i6], E6[i6], F6[i6]),
                               "1000K": (R10[i10], E10[i10], F10[i10])}.items():
            r[tag] = M.evaluate(model, R, E, F)
        g = torch.Generator().manual_seed(7)
        Q = M.random_rotations(800, g)
        Rr = torch.einsum("bij,baj->bai", Q, R3[test]); Fr = torch.einsum("bij,baj->bai", Q, F3[test])
        r["300K_rot"] = M.evaluate(model, Rr, E3[test], Fr)
        r["rot_err_F"], r["rot_err_E"] = M.rotation_error(model, R3[test][:200], torch.Generator().manual_seed(3))
        res[f"{kind}_N{N}"] = r
        print("N=%4d %-9s params %6d | 300K: E %6.1f meV F %5.0f meV/A | 600K F %5.0f | 1000K F %5.0f | "
              "rotated 300K F %5.0f | rot.err F %.1e E %.1e meV | cpu %4.0f s" % (
              N, kind, r["params"], 1e3 * r["300K"]["E_mae"], 1e3 * r["300K"]["F_mae"], 1e3 * r["600K"]["F_mae"],
              1e3 * r["1000K"]["F_mae"], 1e3 * r["300K_rot"]["F_mae"], r["rot_err_F"], 1e3 * r["rot_err_E"], cpu), flush=True)
        json.dump(res, open(path, "w"))
