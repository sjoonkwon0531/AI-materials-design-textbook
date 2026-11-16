"""Lab 7.1.2 -- backprop by hand (NumPy, float64) vs torch autograd;
cost of backward vs forward; gradient flow through a deep MLP.
Run: python 71_backprop.py  (about 1 min on one shared CPU core)"""
import time, numpy as np, torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)

# ---------------- Part A: hand-written backprop for a 2-hidden-layer MLP -------------
def silu(z):  return z / (1 + np.exp(-z))
def dsilu(z):
    s = 1 / (1 + np.exp(-z)); return s * (1 + z * (1 - s))

def init(sizes):
    Ws, bs = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        Ws.append(rng.normal(0, np.sqrt(2.0 / (fan_in + fan_out)), (fan_out, fan_in)))  # Glorot normal
        bs.append(rng.normal(0, 0.1, fan_out))
    return Ws, bs

def forward(Ws, bs, X):
    """X: (d0, B). returns list of pre-activations z and activations a (a[0]=X)"""
    a, z = [X], []
    for l, (W, b) in enumerate(zip(Ws, bs)):
        zl = W @ a[-1] + b[:, None]; z.append(zl)
        a.append(zl if l == len(Ws) - 1 else silu(zl))   # linear output layer
    return z, a

def backward(Ws, z, a, Y):
    """MSE loss L = (1/2B) sum (yhat - y)^2. Returns dL/dW, dL/db (matrix form)."""
    B = Y.shape[1]
    delta = (a[-1] - Y) / B                       # dL/dz_L (output is linear)
    gW, gb = [None] * len(Ws), [None] * len(Ws)
    for l in reversed(range(len(Ws))):
        gW[l] = delta @ a[l].T                    # dL/dW_l = delta_l a_{l-1}^T
        gb[l] = delta.sum(axis=1)
        if l > 0:
            delta = (Ws[l].T @ delta) * dsilu(z[l - 1])   # delta_{l-1} = (W_l^T delta_l) * f'(z_{l-1})
    return gW, gb

sizes = [8, 32, 16, 1]; B = 64
Ws, bs = init(sizes)
X = rng.normal(size=(sizes[0], B)); Y = rng.normal(size=(1, B))
z, a = forward(Ws, bs, X)
loss_np = 0.5 * np.mean((a[-1] - Y) ** 2)
gW, gb = backward(Ws, z, a, Y)

tW = [torch.tensor(W, requires_grad=True) for W in Ws]
tb = [torch.tensor(b, requires_grad=True) for b in bs]
h = torch.tensor(X)
for l in range(len(tW)):
    h = tW[l] @ h + tb[l][:, None]
    if l < len(tW) - 1: h = torch.nn.functional.silu(h)
loss_t = 0.5 * ((h - torch.tensor(Y)) ** 2).mean()
loss_t.backward()
dmax = max(max(np.abs(gW[l] - tW[l].grad.numpy()).max(), np.abs(gb[l] - tb[l].grad.numpy()).max())
           for l in range(len(Ws)))
gmax = max(np.abs(g).max() for g in gW)
print("=== Part A: hand-written backprop vs torch.autograd (float64) ===")
print(f"MLP sizes {sizes}, batch {B}, SiLU hidden, linear output; loss numpy {loss_np:.12f}  torch {loss_t.item():.12f}")
print(f"max |grad_numpy - grad_torch| over all W,b = {dmax:.2e}   (largest gradient entry {gmax:.3f})")
# central finite difference on one weight, for comparison
eps = 1e-6; i, j = 3, 5
Wp = [W.copy() for W in Ws]; Wp[0][i, j] += eps
Wm = [W.copy() for W in Ws]; Wm[0][i, j] -= eps
Lp = 0.5 * np.mean((forward(Wp, bs, X)[1][-1] - Y) ** 2)
Lm = 0.5 * np.mean((forward(Wm, bs, X)[1][-1] - Y) ** 2)
print(f"finite difference dL/dW1[{i},{j}] = {(Lp-Lm)/(2*eps):.10f}   backprop = {gW[0][i,j]:.10f}"
      f"   |diff| = {abs((Lp-Lm)/(2*eps)-gW[0][i,j]):.1e}")

# ---------------- Part B: cost of backward relative to forward ----------------------
print("=== Part B: wall time of forward vs forward+backward (float32, 1 thread) ===")
torch.set_default_dtype(torch.float32)
for width in (256, 1024):
    net = torch.nn.Sequential(*[m for _ in range(4) for m in (torch.nn.Linear(width, width), torch.nn.SiLU())],
                              torch.nn.Linear(width, 1))
    xb = torch.randn(512, width)
    def fwd():
        with torch.no_grad(): net(xb).sum()
    def fwdbwd():
        net.zero_grad(set_to_none=True); net(xb).sum().backward()
    for f in (fwd, fwdbwd): f()
    def best(f, n=30):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
        return min(ts)
    tf, tfb = best(fwd), best(fwdbwd)
    print(f"width {width:5d}: forward {tf*1e3:7.2f} ms   forward+backward {tfb*1e3:7.2f} ms   ratio {tfb/tf:4.2f}")

# ---------------- Part C: gradient flow in a 30-layer MLP ---------------------------
print("=== Part C: per-layer gradient norm ||dL/dW_l|| in a depth-30, width-64 MLP ===")
torch.manual_seed(0)
D, W_, Bc = 30, 64, 256
xc = torch.randn(Bc, W_); yc = torch.randn(Bc, 1)

class Deep(torch.nn.Module):
    def __init__(self, act, init, residual=False, norm=False):
        super().__init__()
        self.lins = torch.nn.ModuleList([torch.nn.Linear(W_, W_) for _ in range(D)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(W_) if norm else torch.nn.Identity() for _ in range(D)])
        self.out = torch.nn.Linear(W_, 1); self.act = act; self.residual = residual
        for lin in self.lins:
            torch.nn.init.zeros_(lin.bias)
            if init == "xavier": torch.nn.init.xavier_normal_(lin.weight)
            elif init == "he": torch.nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            elif init == "small": torch.nn.init.normal_(lin.weight, 0, 0.05)
            elif init == "large": torch.nn.init.normal_(lin.weight, 0, 0.3)
    def forward(self, x):
        for lin, nm in zip(self.lins, self.norms):
            hnew = self.act(lin(nm(x)))
            x = x + hnew if self.residual else hnew
        return self.out(x)

configs = [("sigmoid, Xavier", torch.sigmoid, "xavier", False, False),
           ("tanh, Xavier", torch.tanh, "xavier", False, False),
           ("ReLU, N(0,0.05^2)", torch.relu, "small", False, False),
           ("ReLU, N(0,0.3^2)", torch.relu, "large", False, False),
           ("ReLU, He", torch.relu, "he", False, False),
           ("ReLU, He + residual + LayerNorm", torch.relu, "he", True, True)]
res = {}
for name, act, ini, resid, nm in configs:
    torch.manual_seed(1)
    net = Deep(act, ini, resid, nm)
    loss = ((net(xc) - yc) ** 2).mean(); loss.backward()
    g = np.array([lin.weight.grad.norm().item() for lin in net.lins])
    res[name] = g
    print(f"{name:34s} layer1 {g[0]:9.2e}  layer15 {g[14]:9.2e}  layer30 {g[-1]:9.2e}  ratio first/last {g[0]/g[-1]:9.2e}")
np.savez("71_backprop_gradflow.npz", names=np.array(list(res.keys())), G=np.array(list(res.values())))
