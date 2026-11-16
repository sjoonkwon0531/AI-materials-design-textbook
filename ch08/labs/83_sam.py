"""Ch.8 Sec.8.3: sharpness-aware optimizers and sharpness diagnostics (importable module).

  SAM   : Foret et al., ICLR 2021        eps = rho * g / ||g||_2
  ASAM  : Kwon et al., ICML 2021         eps = rho * T^2 g / ||T g||_2,  T = diag(|w| + eta)
  FSAM  : "Friendly SAM", Li et al., CVPR 2024 (official code nblt/F-SAM):
          d_t = g_t - sigma * m_{t-1},  m_t = lmbda * m_{t-1} + (1 - lmbda) * g_t,  eps = rho * d_t / ||d_t||
          (the batch gradient minus an EMA estimate of the full gradient; first step uses g_1)

Usage (any torch base optimizer):
    opt = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, mode="sam", lr=1e-3, weight_decay=1e-4)
    def closure():
        opt.zero_grad(); loss = loss_fn(); loss.backward(); return loss
    loss = opt.step(closure)        # 2 forward-backward passes per step

Diagnostics:
    lambda_max(loss_fn, params)          largest Hessian eigenvalue: Lanczos (scipy eigsh, which='LA') on Hessian-vector
                                         products (double backward); method='power' = strict power-iteration cross-check
    filter_normalized_direction(model)   Li et al. (NeurIPS 2018) random direction, rescaled per filter (row)
    loss_slice(model, loss_fn, d, alphas)
"""
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, mode="sam", eta=0.01, sigma=1.0, lmbda=0.6, **kw):
        assert mode in ("sam", "asam", "fsam") and rho >= 0
        super().__init__(params, dict(rho=rho, **kw))
        self.base = base_optimizer(self.param_groups, **kw)
        self.param_groups = self.base.param_groups
        self.mode, self.eta, self.sigma, self.lmbda = mode, eta, sigma, lmbda

    @torch.no_grad()
    def first_step(self):
        ps = [p for g in self.param_groups for p in g["params"] if p.grad is not None]
        if self.mode == "fsam":                                  # remove the (EMA) full-gradient component
            for p in ps:
                st = self.state[p]; g = p.grad.clone()
                if "m" in st:
                    p.grad.sub_(self.sigma * st["m"]); st["m"].mul_(self.lmbda).add_(g, alpha=1 - self.lmbda)
                else:
                    st["m"] = g
        if self.mode == "asam":                                  # T_w = |w| + eta (element-wise)
            T = {p: p.abs() + self.eta for p in ps}
            norm = torch.sqrt(sum(((T[p] * p.grad) ** 2).sum() for p in ps))
        else:
            norm = torch.sqrt(sum((p.grad ** 2).sum() for p in ps))
        for g in self.param_groups:
            scale = g["rho"] / (norm + 1e-12)
            for p in g["params"]:
                if p.grad is None: continue
                e = p.grad * scale if self.mode != "asam" else (T[p] ** 2) * p.grad * scale
                self.state[p]["e"] = e
                p.add_(e)                                         # climb to w + eps

    @torch.no_grad()
    def second_step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None or "e" not in self.state[p]: continue
                p.sub_(self.state[p]["e"])                        # back to w, keep grad taken at w + eps
        self.base.step()

    def step(self, closure):
        closure = torch.enable_grad()(closure)
        loss = closure()                                          # grad at w
        self.first_step()
        closure()                                                 # grad at w + eps
        self.second_step()
        return loss


def _flat(ts): return torch.cat([t.reshape(-1) for t in ts])


def lambda_max(loss_fn, params, iters=1000, seed=0, tol=None, method="lanczos", return_vec=False):
    """Largest (algebraic) Hessian eigenvalue of loss_fn() w.r.t. params, from Hessian-vector products (HVPs).
    method="lanczos" (default): scipy.sparse.linalg.eigsh(which="LA", k=1) on a LinearOperator whose matvec is
        one HVP (implicitly restarted Lanczos; relative tol 1e-6, typically some tens of HVPs).
    method="power": plain power iteration with the Rayleigh quotient v^T H v, ||v|| = 1 (Ch.1 Sec.1.5.2), kept as a
        teaching cross-check; it converges to the eigenvalue of largest |lambda| at rate |lambda_2/lambda_1|^k, so it
        is run with a strict stopping rule (relative change < tol = 1e-7 for 5 consecutive steps, up to iters steps).
    For accurate results pass a float64 model/data (HVPs in float32 carry ~1e-6 relative noise).
    Returns (lambda, history): history = Rayleigh quotient of every HVP input (lanczos) or every power step."""
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, eigsh
    tol = tol if tol is not None else (1e-6 if method == "lanczos" else 1e-7)
    params = [p for p in params if p.requires_grad]
    dt = params[0].dtype
    sizes, shapes = [p.numel() for p in params], [p.shape for p in params]
    n = sum(sizes)
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)   # keep graph for the second backward
    hist = []

    def hvp(x):
        v = [t.reshape(s) for t, s in zip(torch.as_tensor(np.asarray(x).reshape(-1), dtype=dt).split(sizes), shapes)]
        Hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        y = _flat(Hv).detach().to(torch.float64).numpy()
        x64 = np.asarray(x, dtype=np.float64).reshape(-1)
        hist.append(float(x64 @ y / (x64 @ x64)))
        return y

    v0 = torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64).numpy()
    if method == "lanczos":
        op = LinearOperator((n, n), matvec=hvp, dtype=np.float64)
        w, V = eigsh(op, k=1, which="LA", tol=tol, v0=v0)
        lam, vec = float(w[0]), V[:, 0]
    elif method == "power":
        v = v0 / np.linalg.norm(v0); lam, streak = 0.0, 0
        for k in range(iters):
            Hv = hvp(v); lam_new = hist[-1]
            streak = streak + 1 if abs(lam_new - lam) < tol * abs(lam_new) else 0
            lam = lam_new; v = Hv / np.linalg.norm(Hv)
            if streak >= 5: break
        else:
            import warnings; warnings.warn(f"power iteration not converged after {iters} steps (lambda ~ {lam:.6g})")
        vec = v
    else:
        raise ValueError(method)
    if return_vec:
        return lam, hist, [torch.from_numpy(t.copy()).reshape(s) for t, s in zip(np.split(vec, np.cumsum(sizes)[:-1]), shapes)]
    return lam, hist


def filter_normalized_direction(model, seed=0):
    """Random Gaussian direction; each output row ('filter') of a weight matrix is rescaled to the norm of
    the corresponding row of the weights. Biases (1-D) get a zero direction, as in Li et al. 2018."""
    g = torch.Generator().manual_seed(seed); d = []
    for p in model.parameters():
        r = torch.randn(p.shape, generator=g)
        if p.dim() <= 1: r.zero_()
        else:
            w2, r2 = p.detach().reshape(p.shape[0], -1), r.reshape(p.shape[0], -1)
            r = (r2 * (w2.norm(dim=1, keepdim=True) / (r2.norm(dim=1, keepdim=True) + 1e-10))).reshape(p.shape)
        d.append(r)
    return d


@torch.no_grad()
def loss_slice(model, loss_fn, d, alphas):
    w0 = [p.detach().clone() for p in model.parameters()]; out = []
    for a in alphas:
        for p, w, di in zip(model.parameters(), w0, d): p.copy_(w + a * di)
        out.append(float(loss_fn()))
    for p, w in zip(model.parameters(), w0): p.copy_(w)
    return out
