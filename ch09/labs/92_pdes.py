"""Lab 9.2-pdes: two nonlinear PDEs with PINNs, each against a classical reference.
(A) viscous Burgers  u_t + u u_x = nu u_xx, x in [-1,1], t in [0,1], u(x,0) = -sin(pi x), u(+-1,t) = 0, nu = 0.01/pi.
    Reference: Cole-Hopf formula with Gauss-Hermite quadrature.  Baseline: conservative central FD + LSODA (banded).
(B) 1D Cahn-Hilliard  c_t = (c^3 - c - eps^2 c_xx)_xx, x in [0, 2 pi) periodic, eps = 0.25, t in [0, T].
    Mixed formulation with the chemical potential mu as a second network output (only 2nd derivatives needed).
    Reference: the stabilized semi-implicit spectral scheme of Ch.2 Derivation Box 2.22 (N = 256, dt = 2e-5).
"""
import math, time, json, sys
import numpy as np, torch
from scipy.integrate import solve_ivp

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
PI = math.pi
quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
only = sys.argv[2] if len(sys.argv) > 2 else "AB"


def mlp(nin, nout, width, depth, seed):
    torch.manual_seed(seed); L, d = [], nin
    for _ in range(depth):
        L += [torch.nn.Linear(d, width), torch.nn.Tanh()]; d = width
    return torch.nn.Sequential(*L, torch.nn.Linear(d, nout))


def grad(y, x):
    return torch.autograd.grad(y.sum(), x, create_graph=True)[0]


def optimize(params, loss_fn, n_adam, n_lbfgs, lr=2e-3, cb=None):
    opt = torch.optim.Adam(params, lr=lr)
    for it in range(n_adam):
        opt.zero_grad(); loss_fn().backward(); opt.step()
    if cb: cb("adam")
    lb = torch.optim.LBFGS(params, max_iter=n_lbfgs, history_size=50, tolerance_grad=1e-12, tolerance_change=1e-15,
                           line_search_fn="strong_wolfe")
    def closure():
        lb.zero_grad(); L = loss_fn(); L.backward(); return L
    lb.step(closure)


out = {}
t_all = time.process_time()
# =============================================================== A. Burgers
if "A" in only:
    nu = 0.01 / PI
    zq, wq = np.polynomial.hermite.hermgauss(120)

    def burgers_exact(x, t):
        """Cole-Hopf solution for u0 = -sin(pi x) (x: array, t > 0 scalar)."""
        eta = math.sqrt(4 * nu * t) * zq[None, :]
        y = x[:, None] - eta
        f = np.exp(-np.cos(PI * y) / (2 * PI * nu))
        return -(np.sin(PI * y) * f) @ wq / (f @ wq)

    xg = np.linspace(-1, 1, 401); tg = np.linspace(0, 1, 101)
    UG = np.array([(-np.sin(PI * xg)) if t == 0 else burgers_exact(xg, t) for t in tg]).T      # (401, 101)
    XG, TG = np.meshgrid(xg, tg, indexing="ij")
    xg_t = torch.tensor(XG.reshape(-1, 1)); tg_t = torch.tensor(TG.reshape(-1, 1))
    xfine = np.linspace(-0.02, 0.02, 4001); tt_ = np.linspace(0.4, 0.6, 201)
    sl = [np.abs(np.gradient(burgers_exact(xfine, t), xfine)).max() for t in tt_]
    print(f"=== A. Burgers, nu = 0.01/pi; reference = Cole-Hopf (120-point Gauss-Hermite); max |u_x| = {max(sl):.2f} at t = {tt_[int(np.argmax(sl))]:.3f} ===")

    def fd_burgers(N):
        x = np.linspace(-1, 1, N + 1); dx = 2 / N
        def rhs(t, u):
            ue = np.r_[0.0, u, 0.0]
            return -(ue[2:] ** 2 - ue[:-2] ** 2) / (4 * dx) + nu * (ue[2:] - 2 * ue[1:-1] + ue[:-2]) / dx**2
        t0 = time.process_time()
        s = solve_ivp(rhs, [0, 1], -np.sin(PI * x[1:-1]), method="LSODA", lband=1, uband=1, t_eval=tg, rtol=1e-8, atol=1e-10)
        w = time.process_time() - t0
        U = np.vstack([np.zeros(len(tg)), s.y, np.zeros(len(tg))])
        ex = np.array([(-np.sin(PI * x)) if t == 0 else burgers_exact(x, t) for t in tg]).T
        return np.linalg.norm(U - ex) / np.linalg.norm(ex), w, s.nfev
    out["A_fd"] = []
    for N in (100, 200, 400, 800, 1600):
        e, w, nf = fd_burgers(N); out["A_fd"].append((N, e, w))
        print(f"  FD N = {N:4d}: rel L2 error {e:.2e}, cpu {w*1e3:6.0f} ms ({nf} RHS evaluations)")

    NF = 2000
    def run_burgers(seed, n_adam, n_lbfgs):
        net = mlp(2, 1, 32, 4, seed)
        u = lambda x, t: -torch.sin(PI * x) + t * (1 - x**2) * net(torch.cat([x, t], 1))      # hard IC and BC
        g = torch.Generator().manual_seed(300 + seed)
        # half of the points uniformly, half concentrated near x = 0 where the front forms
        xf = torch.cat([2 * torch.rand(NF // 2, 1, generator=g) - 1, 0.2 * (2 * torch.rand(NF // 2, 1, generator=g) - 1)])
        tf = torch.rand(NF, 1, generator=g)
        def loss_fn():
            x = xf.clone().requires_grad_(True); t = tf.clone().requires_grad_(True)
            uu = u(x, t); ux = grad(uu, x)
            return (grad(uu, t) + uu * ux - nu * grad(ux, x)).pow(2).mean()
        def err():
            with torch.no_grad():
                up = u(xg_t, tg_t).numpy().reshape(XG.shape)
            return np.linalg.norm(up - UG) / np.linalg.norm(UG), up
        rec = {}
        t0 = time.process_time()
        optimize(list(net.parameters()), loss_fn, n_adam, n_lbfgs, cb=lambda tag: rec.update(adam=err()[0]))
        w = time.process_time() - t0
        e, up = err()
        return e, rec["adam"], w, up, float(loss_fn().detach())
    es, ws = [], []
    for seed in range(2):
        e, ea, w, up, L = run_burgers(seed, *((200, 100) if quick else (1500, 2000)))
        es.append(e); ws.append(w)
        print(f"  PINN (2-32x4-1 tanh, hard IC/BC, {NF} points) seed {seed}: rel L2 after Adam {ea:.2e} -> after L-BFGS {e:.2e}, final loss {L:.1e}, cpu {w:.0f} s")
        if seed == 0: out["A_pinn"] = dict(u=up[:, ::25].tolist(), ref=UG[:, ::25].tolist(), x=xg.tolist(), err=np.abs(up - UG)[::2, ::2].tolist())
    out["A_errs"] = es; out["A_wall"] = ws

# =============================================================== B. Cahn-Hilliard (1D)
if "B" in only:
    eps, T = 0.25, 1.0
    c0_np = lambda x: 0.2 * np.cos(2 * x) + 0.1 * np.cos(3 * x)

    def ch_spectral(N=256, dt=1e-4, S=2.0, nout=101):
        x = np.linspace(0, 2 * PI, N, endpoint=False); k = np.fft.fftfreq(N, d=1.0 / N)
        k2, k4 = k**2, k**4; den = 1 + dt * S * k2 + dt * eps**2 * k4; gain = 1 + dt * S * k2
        c = c0_np(x); ch = np.fft.fft(c); steps = int(round(T / dt)); every = steps // (nout - 1); C = [c.copy()]
        for n in range(1, steps + 1):
            ch = (gain * ch - dt * k2 * np.fft.fft(c**3 - c)) / den; c = np.real(np.fft.ifft(ch))
            if n % every == 0: C.append(c.copy())
        return x, np.array(C).T

    def free_energy(C, x):
        k = np.fft.fftfreq(len(x), d=1.0 / len(x))[:, None]
        cx = np.real(np.fft.ifft(1j * k * np.fft.fft(C, axis=0), axis=0))
        return (0.25 * (C**2 - 1) ** 2 + 0.5 * eps**2 * cx**2).mean(0) * 2 * PI

    t0 = time.process_time(); xs, CS = ch_spectral(dt=2e-5); w_ref = time.process_time() - t0
    t0 = time.process_time(); _, CS2 = ch_spectral(dt=1e-5); w2 = time.process_time() - t0
    tgc = np.linspace(0, T, 101)
    Fs = free_energy(CS, xs)
    print(f"=== B. 1D Cahn-Hilliard, eps = {eps}, T = {T}; spectral reference N=256, dt=2e-5: {w_ref*1e3:.0f} ms ===")
    print(f"  reference self-check: rel L2 diff between dt=2e-5 and dt=1e-5: {np.linalg.norm(CS-CS2)/np.linalg.norm(CS2):.1e}")
    print(f"  reference: max|c| {np.abs(CS[:,0]).max():.2f} -> {np.abs(CS[:,-1]).max():.2f}; mass drift {np.abs(CS.mean(0)-CS[:,0].mean()).max():.1e}; F {Fs[0]:.3f} -> {Fs[-1]:.3f}, increases {int((np.diff(Fs)>1e-12).sum())}/100")
    XS, TS = np.meshgrid(xs, tgc, indexing="ij")
    xs_t = torch.tensor(XS.reshape(-1, 1)); ts_t = torch.tensor(TS.reshape(-1, 1))
    c0_t = lambda x: 0.2 * torch.cos(2 * x) + 0.1 * torch.cos(3 * x)
    NFB = 2000

    def run_ch(seed, n_adam, n_lbfgs, t_scale=True):
        net = mlp(7, 2, 48, 4, seed)
        def cm(x, t):       # periodic embedding (hard periodic BC) + hard IC for c
            z = torch.cat([torch.cos(x), torch.sin(x), torch.cos(2 * x), torch.sin(2 * x), torch.cos(3 * x), torch.sin(3 * x), t / T], 1)
            o = net(z)
            return c0_t(x) + (t / T) * o[:, :1], o[:, 1:]
        g = torch.Generator().manual_seed(400 + seed)
        xf = 2 * PI * torch.rand(NFB, 1, generator=g); tf = T * torch.rand(NFB, 1, generator=g)
        def loss_fn(parts=False):
            x = xf.clone().requires_grad_(True); t = tf.clone().requires_grad_(True)
            c, mu = cm(x, t)
            r1 = grad(c, t) - grad(grad(mu, x), x)
            r2 = mu - (c**3 - c - eps**2 * grad(grad(c, x), x))
            return (r1.pow(2).mean(), r2.pow(2).mean()) if parts else r1.pow(2).mean() + 10.0 * r2.pow(2).mean()
        def diag():
            with torch.no_grad():
                C = cm(xs_t, ts_t)[0].numpy().reshape(XS.shape)
            F = free_energy(C, xs)
            return dict(rel=np.linalg.norm(C - CS) / np.linalg.norm(CS), rel_t=(np.linalg.norm(C - CS, axis=0) / np.linalg.norm(CS, axis=0)),
                        mass=np.abs(C.mean(0) - C[:, 0].mean()).max(), F=F, Finc=int((np.diff(F) > 1e-12).sum()), C=C)
        rec = {}
        t0 = time.process_time()
        optimize(list(net.parameters()), loss_fn, n_adam, n_lbfgs, cb=lambda tag: rec.update(adam=diag()["rel"]))
        w = time.process_time() - t0
        d = diag(); L1, L2 = [float(v.detach()) for v in loss_fn(parts=True)]
        return d, rec["adam"], w, (L1, L2), loss_fn, cm

    for seed in range(1 if quick else 2):
        d, ea, w, Ls, loss_fn, cm = run_ch(seed, *((200, 100) if quick else (1500, 2500)))
        print(f"  PINN mixed (c, mu) seed {seed}: rel L2 after Adam {ea:.2e} -> after L-BFGS {d['rel']:.2e}; err(t=0.25,0.5,1) = {d['rel_t'][25]:.2e}, {d['rel_t'][50]:.2e}, {d['rel_t'][100]:.2e}")
        print(f"       residual losses r1 {Ls[0]:.1e} r2 {Ls[1]:.1e}; mass drift max|<c>(t)-<c>(0)| {d['mass']:.1e}; F {d['F'][0]:.3f} -> {d['F'][-1]:.3f}, increases {d['Finc']}/100; cpu {w:.0f} s")
        if seed == 0:
            out["B"] = dict(x=xs.tolist(), ref=CS[:, ::25].tolist(), c=d["C"][:, ::25].tolist(), rel_t=d["rel_t"].tolist(),
                            mass=np.abs(d["C"].mean(0) - d["C"][:, 0].mean()).tolist(), F=d["F"].tolist(), Fref=Fs.tolist())
            # cost per loss+gradient evaluation: mixed (2nd derivatives of two outputs) vs direct 4th-order residual
            net4 = mlp(7, 1, 48, 4, 0)
            def c_only(x, t):
                z = torch.cat([torch.cos(x), torch.sin(x), torch.cos(2 * x), torch.sin(2 * x), torch.cos(3 * x), torch.sin(3 * x), t / T], 1)
                return c0_t(x) + (t / T) * net4(z)
            g = torch.Generator().manual_seed(1); xf = 2 * PI * torch.rand(NFB, 1, generator=g); tf = T * torch.rand(NFB, 1, generator=g)
            def loss4():
                x = xf.clone().requires_grad_(True); t = tf.clone().requires_grad_(True)
                c = c_only(x, t); cxx = grad(grad(c, x), x)
                return (grad(c, t) - grad(grad(c**3 - c - eps**2 * cxx, x), x)).pow(2).mean()
            def bench(fn, rep=10):
                ts = []
                for _ in range(rep):
                    t0 = time.process_time(); fn().backward(); ts.append(time.process_time() - t0)
                return np.median(ts) * 1e3
            print(f"  cost per loss+gradient ({NFB} points): mixed {bench(loss_fn):.0f} ms, direct 4th-order {bench(loss4):.0f} ms")

print(f"total cpu {time.process_time()-t_all:.0f} s")
if not quick: json.dump(out, open(f"../ch09/data/92_pdes_{only}.json", "w"))
