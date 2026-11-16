"""Lab 9.4-B (part 1): pre-train an ensemble of time-conditioned FNOs on the PDE family
    u_t + c u_x + beta (u^2/2)_x = nu u_xx   (rho = 0 in pre-training; the rho channel is always zero).
Data from 94_data.py.  Run: python 94b_pretrain.py   (about 4 x 60 CPU-s)"""
import sys, time, importlib, numpy as np, torch
sys.path.insert(0, "../ch09/labs")
C = importlib.import_module("94_common")
load = lambda name: dict(np.load(C.D + f"94_{name}.npz"))
WIDTH, MODES, DEPTH, STEPS, BS, LR, MEMBERS = 32, 16, 4, 3000, 32, 3e-3, 4

if __name__ == "__main__":
    tr, te, low, rea, hk = [load(n) for n in ("pre_train", "pre_test", "lownu_test", "react_test", "highk_test")]
    print(f"pre-training set: {tr['u'].shape[0]} trajectories x {tr['u'].shape[1] - 1} times x {tr['u'].shape[2]} points")
    cpu = []
    for k in range(MEMBERS):
        torch.manual_seed(k)
        model = C.TFNO(WIDTH, MODES, DEPTH)
        t0 = time.process_time()
        wall = C.train(model, tr, STEPS, bs=BS, lr=LR, seed=k, log=1000 if k == 0 else None)
        cpu.append(time.process_time() - t0)
        torch.save(model.state_dict(), C.D + f"94_fno_member{k}.pt")
        e = {n: C.traj_error(model, d) for n, d in (("ID", te), ("low-nu", low), ("high-k", hk), ("reaction", rea))}
        print(f"member {k}: {C.n_params(model)} params, {STEPS} steps, {cpu[-1]:.0f} CPU-s (wall {wall:.0f} s) | "
              + " | ".join(f"{n} {100 * v.mean():.2f}%" for n, v in e.items()), flush=True)
    print(f"mean training cost per member: {np.mean(cpu):.0f} CPU-s")
    np.save(C.D + "94_train_cpu.npy", np.array(cpu))
