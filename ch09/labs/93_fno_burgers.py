"""Lab 9.3-1: FNO-1D for the Burgers solution operator u0 -> u(.,T).
(1) train at s = 32/64/128 points, test the SAME weights at 32..512 ("zero-shot super-resolution")
(2) honest baselines: predict on the training grid and Fourier-interpolate; a periodic CNN
(3) discretization self-consistency (aliasing), translation equivariance, mean conservation
(mode-truncation and grid-channel ablations: 93_fno_ablation.py)
Data: data/93_burgers.npz from 93_data.py (nu = 0.005, T = 0.25, fine grid 512).
"""
import os, sys, time, json, importlib
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M = importlib.import_module("93_models")
t_all = time.time(); c_all = time.process_time()

d = np.load(os.path.join(HERE, "..", "data", "93_burgers.npz"))
traj = torch.tensor(d["traj"]); NTR, NTE, NF = 1000, 200, traj.shape[-1]
A, U = traj[:, 0], traj[:, -1]                                # (n, 512): u0 and u(T)
tr, te = slice(0, NTR), slice(2000, 2000 + NTE)
sub = lambda X, s: X[:, None, ::NF // s]                      # point-wise subsampling to s points, add channel
RES = [32, 64, 128, 256, 512]
EPOCHS, BATCH = 50, 20


def train(net, s, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed)
    a, u = sub(A[tr], s), sub(U[tr], s)
    return M.fit(net, lambda idx: M.rel_l2(net(a[idx]), u[idx]), NTR, epochs, BATCH, lr=3e-3, seed=seed)


@torch.no_grad()
def test(net, s):
    return M.rel_l2_each(net(sub(A[te], s)), sub(U[te], s))


out = {"res": RES}
# ---------------- (1) resolution transfer ----------------
print(f"data: {NTR} train / {NTE} test, nu={float(d['nu'])}, T={float(d['T'])}; width 32, depth 4, {EPOCHS} epochs, batch {BATCH}")
nets = {}
for s in (32, 64, 128):
    torch.manual_seed(0)
    net = M.FNO1d(width=32, kmax=16, depth=4); h, wall = train(net, s); nets[s] = net
    e = [test(net, r).mean().item() for r in RES]
    out[f"fno_train{s}"] = e; out[f"wall_train{s}"] = wall
    print(f"FNO(kmax=16) trained at s={s:3d} ({wall:4.0f} s): test rel-L2 at s=" + "  ".join(f"{r}: {x:.4f}" for r, x in zip(RES, e)))
print(f"FNO parameters: {M.count_params(nets[64])}")

# ---------------- (2) honest baselines ----------------
with torch.no_grad():
    u512 = U[te][:, None]
    for s in (32, 64, 128):
        p = M.fourier_interp(nets[s](sub(A[te], s)), NF)                  # predict coarse, then interpolate
        tru = M.fourier_interp(sub(U[te], s), NF)                         # interpolating the TRUE coarse solution
        out[f"interp_pred{s}"] = M.rel_l2(p, u512).item(); out[f"interp_true{s}"] = M.rel_l2(tru, u512).item()
        print(f"  s={s:3d} -> 512: zero-shot {out[f'fno_train{s}'][-1]:.4f} | FNO@{s} + Fourier interp {out[f'interp_pred{s}']:.4f}"
              f" | interp of exact coarse solution (floor) {out[f'interp_true{s}']:.4f}")
    # spectrum of the s=64 model evaluated at 512 vs truth
    ph = torch.fft.rfft(nets[64](sub(A[te], 512))[:, 0]).abs().pow(2).mean(0).sqrt() / NF
    th = torch.fft.rfft(U[te]).abs().pow(2).mean(0).sqrt() / NF
    eh = torch.fft.rfft(nets[64](sub(A[te], 512))[:, 0] - U[te]).abs().pow(2).mean(0).sqrt() / NF
    out["spec_true"], out["spec_pred64"], out["spec_err64"] = th.tolist(), ph.tolist(), eh.tolist()
    for kk in (8, 24, 40, 64, 96):
        print(f"  rms |u_k| at k={kk:3d}: truth {th[kk]:.2e}  FNO(train 64, eval 512) {ph[kk]:.2e}  error {eh[kk]:.2e}")

    up = lambda s: M.fourier_interp(sub(A[te], s), NF)                    # coarse input, upsampled: no new information
    out["coarse_in32"] = M.rel_l2(nets[32](up(32)), u512).item(); out["coarse_in64"] = M.rel_l2(nets[64](up(64)), u512).item()
    print(f"  evaluate at 512 but feed the upsampled COARSE input: train 32: {out['coarse_in32']:.4f} | train 64: {out['coarse_in64']:.4f}")

torch.manual_seed(0)
cnn = M.CNN1d(width=32, ksize=33, depth=4); h, wall = train(cnn, 64)
e = [test(cnn, r).mean().item() for r in RES]; out["cnn_train64"] = e; out["cnn_params"] = M.count_params(cnn)
print(f"CNN(kernel 33 points, {M.count_params(cnn)} params) trained at s=64 ({wall:.0f} s): " + "  ".join(f"{r}: {x:.4f}" for r, x in zip(RES, e)))

# ---------------- (3) structure checks ----------------
with torch.no_grad():
    net = nets[64]
    lo = net(sub(A[te], 64))
    for s in (128, 512):                                                  # same function, two grids
        hi = net(sub(A[te], s))[..., ::s // 64]
        out[f"selfcons_{s}"] = M.rel_l2(hi, lo).item()
    abl = M.fourier_interp(M.fourier_interp(A[te][:, None], 16), NF)      # band-limited input (|k| < 8): no input aliasing
    ref = net(abl)
    out["selfcons_bl"] = [M.rel_l2(net(abl[..., ::NF // s]), ref[..., ::NF // s]).item() for s in (32, 64, 128)]
    print(f"self-consistency of FNO(train 64): ||G(a_s)|_64 - G(a_64)|| / ||G(a_64)||  s=128: {out['selfcons_128']:.4f}  s=512: {out['selfcons_512']:.4f}")
    print("  band-limited input (|k|<8), grid s vs grid 512: " + "  ".join(f"s={s}: {x:.1e}" for s, x in zip((32, 64, 128), out["selfcons_bl"])))
    a64 = sub(A[te], 64); sh = 5
    eq = M.rel_l2(net(torch.roll(a64, sh, -1)), torch.roll(net(a64), sh, -1)).item()
    out["equiv_nogrid"] = eq
    dm = (net(a64).mean(-1) - a64.mean(-1)).abs()[:, 0]
    out["mass_err_mean"], out["mass_err_max"] = dm.mean().item(), dm.max().item()
    print(f"mean conservation |mean(pred) - mean(u0)|: mean {dm.mean():.2e}, max {dm.max():.2e}  (solver: 1e-15)")
# ---------------- (4) cost ----------------
with torch.no_grad():
    a64 = sub(A[te], 64); t0 = time.time()
    for _ in range(20): nets[64](a64)
    t_inf = (time.time() - t0) / 20 / NTE
D = importlib.import_module("93_data")
t0 = time.time(); D.burgers_solve(A[te].numpy().astype(float), 0.005, 0.25, 5e-4); t_sol = (time.time() - t0) / NTE
out.update(t_inf=t_inf, t_sol=t_sol)
print(f"cost per sample: FNO forward (s=64) {t_inf*1e3:.3f} ms | reference solver N=512, dt=5e-4: {t_sol*1e3:.1f} ms")
Un = U[te].numpy().astype(float); An = A[te].numpy().astype(float)
for (Nc, dtc) in ((64, 2.5e-3), (96, 1.25e-3), (128, 1.25e-3)):          # the same solver on coarse grids
    ac = M.fourier_interp(torch.tensor(An), Nc).numpy() if NF % Nc else An[:, ::NF // Nc]
    t0 = time.time(); uc = D.burgers_solve(ac, 0.005, 0.25, dtc); tc = (time.time() - t0) / NTE
    uf = M.fourier_interp(torch.tensor(uc), NF).numpy()
    ec = float(np.mean(np.linalg.norm(uf - Un, axis=1) / np.linalg.norm(Un, axis=1)))
    out[f"sol{Nc}"] = dict(t=tc, err=ec, dt=dtc)
    print(f"   coarse solver N={Nc:3d}, dt={dtc:.2e}: {tc*1e3:.2f} ms, rel-L2 at 512 (after Fourier interpolation) {ec:.4f}")
with torch.no_grad():
    j = int(torch.argsort(test(nets[64], 512))[NTE // 2])                 # median test sample
    out["example"] = dict(a=A[te][j].tolist(), u=U[te][j].tolist(), p64=nets[64](sub(A[te], 512))[j, 0].tolist(),
                          p32=nets[32](sub(A[te], 512))[j, 0].tolist(), cnn=cnn(sub(A[te], 512))[j, 0].tolist(), cnn64=cnn(sub(A[te], 64))[j, 0].tolist())
json.dump(out, open(os.path.join(HERE, "93_fno_burgers.json"), "w"))
torch.save({s: n.state_dict() for s, n in nets.items()}, os.path.join(HERE, "93_fno_burgers.pt"))
print(f"wall {time.time()-t_all:.0f} s (CPU {time.process_time()-c_all:.0f} s)")
