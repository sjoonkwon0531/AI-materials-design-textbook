"""Case Study 9.6, evaluation: what does the FNO surrogate of Cahn-Hilliard preserve?
Three models trained by 96_fno_ch_train.py on the same data:
    free    : plain FNO, one-step loss
    cons    : zero-mode constraint (mass conserved by construction), one-step loss
    cons_pf : zero-mode constraint, pushforward (two-step) loss
The solver takes the first big step (the quench, t = 0 -> 2.5); the surrogate is rolled out
autoregressively from the true c(2.5) for 19 big steps (Dt = 2.5) up to t = 50.
Measured: one-step and rollout error, mass, free-energy decay, growth law, symmetries,
out-of-distribution mean compositions, and the cost. Inference in float64, one thread."""
import re, time, importlib.util
import numpy as np
import torch

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
ch, fno = load("ch", "96_ch_solver.py"), load("fno", "95_fno2d.py")
torch.set_num_threads(1)

d = np.load("../data/96_ch_64.npz")
test, ood, m_ood = d["test"].astype(float), d["ood"].astype(float), d["m_ood"]
NREC = test.shape[1] - 1                                   # 20 big steps
t = 2.5 * np.arange(NREC + 1)
sol = ch.CH2D(64)
rel = lambda a, b: np.sqrt(((a - b) ** 2).sum((-2, -1)) / (b**2).sum((-2, -1)))
NAMES = {"free": "FNO, one-step loss", "cons": "FNO + zero mode, one-step loss", "cons_pf": "FNO + zero mode, pushforward loss"}

models, info = {}, {}
for tag in NAMES:
    ck = torch.load(f"../data/96_fno_ch_{tag}.pt", weights_only=False)
    m = fno.FNO2d(**ck["config"]); m.load_state_dict(ck["state_dict"]); m.eval()
    models[tag], info[tag] = fno.to_double(m), ck

def apply(model, x, bs=20):
    """One surrogate step on a numpy batch (N, 64, 64), in chunks to keep the memory small."""
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(x[i:i + bs])) for i in range(0, len(x), bs)]).numpy()

def one_step(model, traj):                                 # teacher forcing: true c(t) -> c(t + Dt)
    return apply(model, traj[:, :-1].reshape(-1, 64, 64)).reshape(traj[:, 1:].shape)

def rollout(model, c0, n):
    out = [c0]
    for _ in range(n):
        out.append(apply(model, out[-1]))
    return np.stack(out, 1)

def diagnostics(traj):
    F = np.stack([sol.free_energy(traj[:, i]) for i in range(traj.shape[1])], 1)
    L = np.stack([sol.length(traj[:, i]) for i in range(traj.shape[1])], 1)
    drift = np.abs(traj.mean((2, 3)) - traj[:, :1].mean((2, 3)))
    return F, L, drift

def slope(tt, L, lo=5.0, hi=50.0):
    f = (tt >= lo) & (tt <= hi)
    return np.polyfit(np.log(tt[f]), np.log(L[f]), 1)[0]

t1 = t[1:]                                                 # times of a rollout started at t = 2.5
F_s, L_s, _ = diagnostics(test[:, 1:])
pers = rel(test[:, :-1], test[:, 1:])
frozen = rel(np.repeat(test[:, 1:2], NREC - 1, 1), test[:, 2:])        # "nothing happens after t = 2.5"
save = dict(t=t, F_solver=F_s, L_solver=L_s, pers=pers, frozen=frozen)
print("=== in-distribution test set: 40 trajectories, |m| <= 0.2 ===")
print("baselines:  persistence (one step) 0->2.5: %.3f | 2.5->5: %.3f | 22.5->25: %.3f | 47.5->50: %.3f | mean of steps 2..20: %.3f"
      % (pers[:, 0].mean(), pers[:, 1].mean(), pers[:, 9].mean(), pers[:, 19].mean(), pers[:, 1:].mean()))
print("            frozen at c(2.5):      t=5: %.3f | t=25: %.3f | t=50: %.3f" % tuple(frozen[:, i].mean() for i in (0, 8, 18)))
print(f"solver:     F(2.5) {F_s[:, 0].mean():.3f} -> F(50) {F_s[:, -1].mean():.3f}, increases 0 / {F_s[:, 1:].size};"
      f" L(5) {L_s[:, 1].mean():.3f}, L(25) {L_s[:, 9].mean():.3f}, L(50) {L_s[:, -1].mean():.3f},"
      f" slope on 5 <= t <= 50: {slope(t1, L_s.mean(0)):.3f}")
roll = {}
for tag, model in models.items():
    p1 = one_step(model, test); e1 = rel(p1, test[:, 1:])
    r = rollout(model, test[:, 1], NREC - 1); er = rel(r[:, 1:], test[:, 2:]); roll[tag] = r
    r0 = rollout(model, test[:, 0], NREC); er0 = rel(r0[:, 1:], test[:, 1:])          # surrogate also takes the quench step
    F, L, drift = diagnostics(r)
    dF = np.diff(F, axis=1)
    F_in = np.stack([sol.free_energy(test[:, i]) for i in range(1, NREC)], 1)       # one-step energy check, true inputs
    F_out = np.stack([sol.free_energy(p1[:, i]) for i in range(1, NREC)], 1)
    x = test[:, 1:].reshape(-1, 64, 64)[::8]               # 100 states for the symmetry checks
    gx = apply(model, x)
    odd = rel(-apply(model, -x), gx).mean()
    shift = lambda a: np.roll(a, (17, 5), axis=(-2, -1))
    equi = rel(apply(model, shift(x)), shift(gx)).mean()
    inc = np.abs((p1 - test[:, :-1]).mean((2, 3)))
    print(f"\n--- {tag}: {NAMES[tag]} (training CPU time {info[tag]['cpu']:.0f} s) ---")
    print("  one-step error (true input):   0->2.5: %.4f | 2.5->5: %.4f | 22.5->25: %.4f | 47.5->50: %.4f | mean of steps 2..20: %.4f"
          % (e1[:, 0].mean(), e1[:, 1].mean(), e1[:, 9].mean(), e1[:, 19].mean(), e1[:, 1:].mean()))
    print("  rollout from the true c(2.5):  t=5: %.4f | t=12.5: %.4f | t=25: %.4f | t=50: %.4f" % tuple(er[:, i].mean() for i in (0, 3, 8, 18)))
    print("  rollout from c(0) (surrogate takes the quench step too): t=2.5: %.3f | t=25: %.3f | t=50: %.3f; max |c| = %.2f"
          % (er0[:, 0].mean(), er0[:, 9].mean(), er0[:, 19].mean(), np.abs(r0).max()))
    print(f"  [1] mass: max |mean(t) - mean(2.5)| over 40 rollouts x 19 steps = {drift.max():.2e}; mean drift at t = 50: {drift[:, -1].mean():.2e};"
          f" mean |mean of one-step increment| = {inc.mean():.2e}")
    print(f"  [2] free energy: increases along the rollouts {int((dF > 0).sum())} / {dF.size};"
          f" one step from true states: F(G(c)) > F(c) in {int((F_out > F_in).sum())} / {F_in.size};"
          f" F(50) = {F[:, -1].mean():.3f} (solver {F_s[:, -1].mean():.3f})")
    print(f"  [3] growth law: L(25) = {L[:, 9].mean():.3f}, L(50) = {L[:, -1].mean():.3f} (solver {L_s[:, 9].mean():.3f}, {L_s[:, -1].mean():.3f});"
          f" slope on 5 <= t <= 50: {slope(t1, L.mean(0)):.3f} (solver {slope(t1, L_s.mean(0)):.3f})")
    print(f"  bounds: max |c| = {np.abs(r).max():.3f} (solver {np.abs(test).max():.3f});"
          f"  symmetries: ||G(-c) + G(c)|| / ||G(c)|| = {odd:.4f},  ||G(Sc) - S G(c)|| / ||G(c)|| = {equi:.4f}")
    save.update({f"e1_{tag}": e1, f"er_{tag}": er, f"er0_{tag}": er0, f"F_{tag}": F, f"L_{tag}": L, f"drift_{tag}": drift})

xs = test[:8, 1:].reshape(-1, 64, 64)[::4]
gs = sol.advance(xs, 50)
print(f"\nsolver, same symmetry checks: odd {rel(-sol.advance(-xs, 50), gs).mean():.1e},"
      f" translation {rel(sol.advance(np.roll(xs, (17, 5), axis=(-2, -1)), 50), np.roll(gs, (17, 5), axis=(-2, -1))).mean():.1e}")
k, S_s = sol.structure_factor(test[:, -1])
save.update(k=k, S_solver=S_s, **{f"S_{tag}": sol.structure_factor(roll[tag][:, -1])[1] for tag in models},
            snap_solver=test[:2, [2, 5, 10, 20]], **{f"snap_{tag}": roll[tag][:2, [1, 4, 9, 19]] for tag in models})

print("\n=== out of distribution: mean composition outside the training range, 16 trajectories each ===")
print("  |m|   model     one-step (steps 2..20)   rollout from c(2.5): t=5   t=25    t=50    mass drift   F increases   L(50) model / solver")
for mm in (0.3, 0.4, 0.5):
    sel = np.abs(m_ood) == mm
    Lo = sol.length(ood[sel][:, -1]).mean()
    fz = rel(np.repeat(ood[sel][:, 1:2], NREC - 1, 1), ood[sel][:, 2:])
    print(f"  {mm:.1f}   frozen at c(2.5)                                       {fz[:, 0].mean():.4f}   {fz[:, 8].mean():.4f}  {fz[:, 18].mean():.4f}")
    for tag, model in models.items():
        e1 = rel(one_step(model, ood[sel]), ood[sel][:, 1:])
        r = rollout(model, ood[sel][:, 1], NREC - 1); er = rel(r[:, 1:], ood[sel][:, 2:])
        F, L, drift = diagnostics(r)
        print(f"  {mm:.1f}   {tag:<8s}  {e1[:, 1:].mean():.4f}                                 {er[:, 0].mean():.4f}   {er[:, 8].mean():.4f}  {er[:, 18].mean():.4f}"
              f"  {drift.max():.1e}      {int((np.diff(F, axis=1) > 0).sum()):3d} / {F[:, 1:].size}     {L[:, -1].mean():.2f} / {Lo:.2f}")
        save.update({f"ood_er_{tag}_{mm}": er, f"ood_e1_{tag}_{mm}": e1})
        if tag == "free":                                  # where does the unconstrained model move the mean?
            ms = m_ood[sel]; mean50 = r[:, -1].mean((1, 2)); mean0 = ood[sel][:, 1].mean((1, 2))
            note = (f"        free model, mean composition at t = 50: {mean50[ms > 0].mean():+.3f} for trajectories started at"
                    f" {mean0[ms > 0].mean():+.3f}, {mean50[ms < 0].mean():+.3f} for those started at {mean0[ms < 0].mean():+.3f}")
        if tag == "cons_pf":
            save[f"ood_snap_{mm}"] = np.stack([ood[sel][0, [1, 2, 20]], r[0, [0, 1, 19]]])
    print(note)
for tag in models:                                         # error as a function of |m| inside the training range
    e = save[f"er_{tag}"][:, 8]
    am = np.abs(d["m_test"])
    print(f"  inside the range, {tag}: rollout error at t = 25 for |m| < 0.1: {e[am < 0.1].mean():.4f} ({int((am < 0.1).sum())} traj.),"
          f" for 0.1 <= |m| <= 0.2: {e[am >= 0.1].mean():.4f} ({int((am >= 0.1).sum())} traj.)")

print("\n=== cost: one thread, same process, smallest CPU time of 20 calls ===")
def best(f, rep=20):
    """Smallest CPU time of one call (process_time, so that other jobs on the machine are not counted)."""
    b = np.inf
    for _ in range(rep):
        t0 = time.process_time(); f(); b = min(b, time.process_time() - t0)
    return b
m32 = fno.FNO2d(**info["cons_pf"]["config"]); m32.load_state_dict(info["cons_pf"]["state_dict"]); m32.eval()
x1 = torch.from_numpy(test[:1, 4].astype(np.float32)); x40 = torch.from_numpy(test[:, 4].astype(np.float32))
with torch.no_grad():
    ms_fno, ms_fno40 = 1e3 * best(lambda: m32(x1)), 1e3 * best(lambda: m32(x40), 5) / 40
c1, c40 = test[:1, 4], test[:, 4]
ms_sol, ms_sol40 = 1e3 * best(lambda: sol.advance(c1, 50)), 1e3 * best(lambda: sol.advance(c40, 50), 5) / 40
gen_s = float(re.search(r"CPU ([\d.]+) s", open("96_ch_data.out").read()).group(1))
train_s = float(info["cons_pf"]["cpu"])
print(f"one big step (Dt = 2.5): FNO {ms_fno:.2f} ms (batch 1), {ms_fno40:.2f} ms/sample (batch 40);"
      f" solver, 50 steps: {ms_sol:.2f} ms (batch 1), {ms_sol40:.2f} ms/sample (batch 40)")
print(f"speed-up per big step: {ms_sol / ms_fno:.2f}x (batch 1), {ms_sol40 / ms_fno40:.2f}x (batch 40)")
print(f"one-off cost (CPU time): data {gen_s:.0f} s + training {train_s:.0f} s = {gen_s + train_s:.0f} s", end="; ")
if ms_sol > ms_fno:
    print(f"break-even after {(gen_s + train_s) / (19 * (ms_sol - ms_fno) / 1e3):,.0f} trajectories of 19 big steps")
else:
    print("no break-even: the solver is faster per big step")

print("\nsame scheme with a larger time step, started from the true c(2.5); error against the dt = 0.05 data:")
print("   method            one big step (steps 2..20)   rollout t=5   t=25     t=50     L(50)    slope    F increases   mass drift   ms per Dt (1 sample)")
big = []
for dt in (0.05, 0.1, 0.25, 0.5, 1.25, 2.5):
    s2 = ch.CH2D(64, dt=dt); n = int(round(2.5 / dt))
    e1 = rel(s2.advance(test[:, 1:-1].reshape(-1, 64, 64), n).reshape(test[:, 2:].shape), test[:, 2:])
    tr = s2.trajectory(test[:, 1], NREC - 1, n)
    L = np.stack([sol.length(tr[:, i]) for i in range(NREC)], 1)
    ms = 1e3 * best(lambda: s2.advance(c1, n))
    e = rel(tr[:, 1:], test[:, 2:])
    Fb, _, db = diagnostics(tr)
    big.append((dt, n, e1.mean(), e[:, 0].mean(), e[:, 8].mean(), e[:, 18].mean(), L[:, -1].mean(), slope(t1, L.mean(0)), ms))
    print(f"   solver dt={dt:<5g}   {big[-1][2]:.4f}                       {big[-1][3]:.4f}        {big[-1][4]:.4f}   {big[-1][5]:.4f}   {big[-1][6]:.3f}    {big[-1][7]:.3f}"
          f"    {int((np.diff(Fb, axis=1) > 0).sum()):3d} / {Fb[:, 1:].size}     {db.max():.1e}      {ms:.2f}")
for tag in models:
    e, L = save[f"er_{tag}"], save[f"L_{tag}"]
    print(f"   {tag:<8s}          {save[f'e1_{tag}'][:, 1:].mean():.4f}                       {e[:, 0].mean():.4f}        {e[:, 8].mean():.4f}   {e[:, 18].mean():.4f}"
          f"   {L[:, -1].mean():.3f}    {slope(t1, L.mean(0)):.3f}"
          f"    {int((np.diff(save[f'F_{tag}'], axis=1) > 0).sum()):3d} / {save[f'F_{tag}'][:, 1:].size}     {save[f'drift_{tag}'].max():.1e}      {ms_fno:.2f}")
save.update(big=np.array(big), ms_fno=ms_fno, ms_sol=ms_sol, ms_fno40=ms_fno40, ms_sol40=ms_sol40, gen_s=gen_s, train_s=train_s)
np.savez("../data/96_fno_ch_results.npz", **save)
