"""Ex 9.2.6: (a) collocation points concentrated at early times, (b) causal weighting, beta = 30."""
import importlib.util, sys, time, math, json
import numpy as np, torch
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("conv", "../ch09/labs/92_convection.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
torch.set_default_dtype(torch.float64)
PI2 = 2 * math.pi; BETA = 30; NA, NL = 2400, 600

def points_early(seed, n_f=1000, n_i=200, n_b=100):
    g = torch.Generator().manual_seed(500 + seed)
    xf = PI2 * torch.rand(n_f, 1, generator=g); tf = torch.rand(n_f, 1, generator=g) ** 2          # t = xi^2
    xi = PI2 * torch.rand(n_i, 1, generator=g); tb = torch.rand(n_b, 1, generator=g)
    return xf, tf, xi, tb

def report(tag, model, lf, cpu, extra=""):
    e, et, up = C.evaluate(model, BETA); Lr, Li, Lb = [float(v.detach()) for v in lf(parts=True)]
    amp = np.sqrt(2 * (up**2).mean(0))
    print(f"{tag}: rel L2 {e:.4f} | err(t=0.1) {et[10]:.3f} err(t=0.5) {et[50]:.3f} err(t=1) {et[-1]:.3f} | L_r {Lr:.1e} L_ic {Li:.1e} L_bc {Lb:.1e} | amplitude t=0,0.5,1: {amp[0]:.3f} {amp[50]:.3f} {amp[-1]:.3f} | cpu {cpu:.0f} s{extra}", flush=True)
    return e

def causal_run(seed, eps, M=20):
    model = C.Net(seed, periodic=PERIODIC); pts = C.points(seed); xf, tf, xi, tb = pts
    b = torch.clamp((tf.squeeze() * M).long(), max=M - 1); oh = torch.nn.functional.one_hot(b, M).double(); cnt = oh.sum(0)
    st = {"w": torch.ones(M), "frozen": False}
    def loss_fn(parts=False):
        x = xf.clone().requires_grad_(True); t = tf.clone().requires_grad_(True)
        u = model(x, t)
        ut = torch.autograd.grad(u.sum(), t, create_graph=True)[0]; ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        r2 = (ut + BETA * ux).pow(2).squeeze()
        Lbin = (oh.T @ r2) / cnt                                                      # residual loss of each time slab
        if not st["frozen"]:
            with torch.no_grad(): st["w"] = torch.exp(-eps * (torch.cumsum(Lbin, 0) - Lbin))   # w_i = exp(-eps sum_{k<i} L_k)
        st["L"] = Lbin.detach()
        Li = (model(xi, torch.zeros_like(xi)) - torch.sin(xi)).pow(2).mean()
        Lb = (model(torch.zeros_like(tb), tb) - model(torch.full_like(tb, PI2), tb)).pow(2).mean()
        if parts: return r2.mean(), Li, Lb
        return (st["w"] * Lbin).mean() + Li + Lb
    hist = []
    def snap(it):
        loss_fn(); w = st["w"]; e, et, _ = C.evaluate(model, BETA)
        hist.append((it, int((w > 0.5).sum()), float(w[4]), float(w[9]), float(w[19]), float(st["L"][0]), float(st["L"][9]), float(st["L"][19]), e))
    c0 = time.process_time()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(NA):
        if it in (0, 100, 300, 600, 1200, 1800, 3000, 4500, 6000, 7500): snap(it)
        opt.zero_grad(); loss_fn().backward(); opt.step()
    snap(NA)
    for c in range(NL // 100):                                                        # L-BFGS in chunks of 100, weights refreshed between chunks
        st["frozen"] = False; loss_fn(); st["frozen"] = True
        lb = torch.optim.LBFGS(model.parameters(), max_iter=100, history_size=50, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
        def closure():
            lb.zero_grad(); L = loss_fn(); L.backward(); return L
        lb.step(closure)
        st["frozen"] = False; snap(NA + (c + 1) * 100)
    cpu = time.process_time() - c0
    tag = "causal + periodic embedding" if PERIODIC else "causal"
    e = report(f"{tag} eps={eps:<5g} seed {seed}", model, loss_fn, cpu)
    print("   iter: #slabs with w>0.5 | w_5 w_10 w_20 | L_1 L_10 L_20 | rel L2")
    for h in hist: print(f"   {h[0]:5d}: {h[1]:2d} | {h[2]:.2e} {h[3]:.2e} {h[4]:.2e} | {h[5]:.1e} {h[6]:.1e} {h[7]:.1e} | {h[8]:.3f}")
    return e

PERIODIC = len(sys.argv) > 6 and sys.argv[6] == "periodic"
part = sys.argv[1]
if len(sys.argv) > 4: NA = int(sys.argv[4])
if len(sys.argv) > 5: NL = int(sys.argv[5])
if part == "a":
    for seed in (0, 1):
        pts = points_early(seed); tf = pts[1]
        model = C.Net(seed); lf = C.make_loss(model, BETA, pts); c0 = time.process_time(); C.optimize(model, lf, NA, NL)
        report(f"early-time points (t = xi^2) seed {seed} [fraction of points with t<0.25: {float((tf<0.25).double().mean()):.2f}, t>0.75: {float((tf>0.75).double().mean()):.2f}]", model, lf, time.process_time() - c0)
elif part == "b":
    for eps in [float(a) for a in sys.argv[2].split(",")]:
        for seed in [int(a) for a in sys.argv[3].split(",")]:
            causal_run(seed, eps)
