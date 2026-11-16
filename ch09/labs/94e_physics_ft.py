"""Lab 9.4-B (part 3): adaptation to the reaction family WITHOUT labels.
The equation is known, so unlabeled inputs are free: draw fresh (u0, nu, c, beta, rho) every step and minimise
the PDE residual of the pre-trained model (PINO-style physics fine-tuning, cf. §9.3.4), optionally together with
the data loss on K labelled trajectories.  Run: python 94e_physics_ft.py   (about 3 CPU-min)"""
import sys, time, importlib, numpy as np, torch
sys.path.insert(0, "../ch09/labs")
C = importlib.import_module("94_common"); P = importlib.import_module("94b_pretrain"); FS = importlib.import_module("94c_fewshot")
STEPS, BS = 400, 32


def residual_loss(model, u0, coef, t):
    """Mean squared PDE residual; d/dt by forward-mode AD, the result is back-propagated to the weights."""
    u, ut = torch.func.jvp(lambda tt: model(u0, coef, tt), (t,), (torch.ones_like(t),))
    return ((ut - C.rhs_torch(u, coef)) ** 2).mean()


def physics_finetune(model, K_sub=None, steps=STEPS, lr=1e-3, seed=0, w_data=1.0, scale=1.0):
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    model.train()
    for s in range(steps):
        u0 = torch.from_numpy(C.sample_ic(BS, 128, rng).astype(np.float32))      # unlabeled inputs: no solver call
        coef = torch.from_numpy(np.stack(C.sample_coeffs(BS, rng, "react"), 1).astype(np.float32))
        t = torch.rand(BS, generator=g)
        loss = scale * residual_loss(model, u0, coef, t)
        if K_sub is not None:                                                      # + data loss on K labelled trajectories
            u = torch.from_numpy(K_sub["u"]); cf = torch.from_numpy(K_sub["coef"])
            i = torch.randint(0, len(u), (BS,), generator=g); j = torch.randint(1, 41, (BS,), generator=g)
            loss = loss + w_data * C.rel_l2(model(u[i, 0], cf[i], j.float() / 40), u[i, j]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval(); return model


if __name__ == "__main__":
    pool, te, te_id = [P.load(n) for n in ("react_train", "react_test", "pre_test")]
    base = FS.pretrained()
    r0 = C.residual_norms(base, te).mean().item()
    print(f"pretrained (zero-shot): target-test error {100 * C.traj_error(base, te).mean():.2f} %, rms residual {r0:.3f}")
    for name, kw in [("residual only, 400 steps (0 labels)", dict()),
                     ("residual only, 1600 steps (0 labels)", dict(steps=1600)),
                     ("residual + K = 16 labels, 400 steps", dict(K=16))]:
        errs, ress, cpus = [], [], []
        for s in (0, 1, 2):
            torch.manual_seed(2000 + s); m = FS.pretrained(); sub = None
            if "K" in kw:
                idx = np.random.default_rng(10 * kw["K"] + s).choice(len(pool["u"]), kw["K"], replace=False)
                sub = dict(u=pool["u"][idx], coef=pool["coef"][idx])
            t0 = time.process_time(); physics_finetune(m, sub, steps=kw.get("steps", STEPS), seed=s); cpus.append(time.process_time() - t0)
            errs.append(100 * C.traj_error(m, te).mean().item()); ress.append(C.residual_norms(m, te).mean().item())
            if s == 0 and "1600" in name:
                np.save(C.D + "94e_example.npy", C.predict(m, dict(u=te["u"][:6], coef=te["coef"][:6]), [20, 40]).numpy())
                eid = 100 * C.traj_error(m, te_id).mean().item()
        print(f"  {name:38s} target-test error {np.mean(errs):6.2f} +- {np.std(errs):4.2f} %   rms residual {np.mean(ress):.3f}"
              f"   ({np.mean(cpus):.0f} CPU-s per run)", flush=True)
    print(f"  ID-test error after 1600 residual-only steps (seed 0): {eid:.2f} %")
