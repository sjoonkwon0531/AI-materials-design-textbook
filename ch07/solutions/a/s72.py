import math, numpy as np, torch, sys
torch.set_num_threads(1)
print("=== 7.2.1")
N = 6; k = {-1: 1, 0: -2, 1: 1}
A = np.zeros((N, N), int)
for n in range(N):
    for m, kv in k.items(): A[n, (n+m) % N] += kv
T = np.zeros((N, N), int)
for n in range(N): T[n, (n-1) % N] = 1          # (T x)_n = x_{n-1}
print(A); print("AT==TA:", (A@T == T@A).all()); print("AT=\n", A@T)
ev = np.linalg.eigvals(A); print("eig:", np.round(sorted(ev.real), 6))
for j in range(6):
    u = np.exp(2j*np.pi*j*np.arange(6)/6); lam = (A@u)[0]/u[0]
    print(f"  j={j}: lambda={lam.real:+.4f}, -4 sin^2(pi j/6)={-4*np.sin(np.pi*j/6)**2:+.4f}, resid={np.abs(A@u-lam*u).max():.1e}")
S = np.zeros((3, 6), int); S[0, 0] = S[1, 2] = S[2, 4] = 1
x = np.zeros(6, int); x[0] = 1
print("SAx =", S@A@x, " SATx =", S@A@T@x)
T3 = np.zeros((3, 3), int)
for n in range(3): T3[n, (n-1) % 3] = 1
print("SAT2 == T3 SA:", (S@A@T@T == T3@S@A).all())
print("=== 7.2.2")
for kk in [7]:
    for B in [1, 2, 3, 4, 5]:
        print(f"  k={kk} B={B}: R={1+(kk-1)*(2**B-1)+(2**B-1)} channels = {0.1*(1+kk*(2**B-1)):.1f} deg")
# brute-force check with torch: receptive field via gradient
for B in [3, 5]:
    layers = []
    for b in range(B): layers += [torch.nn.Conv1d(1, 1, 7, padding=3, bias=False), torch.nn.MaxPool1d(2)]
    net = torch.nn.Sequential(*layers)
    for l in net:
        if isinstance(l, torch.nn.Conv1d): torch.nn.init.constant_(l.weight, 1.0)
    x = torch.ones(1, 1, 1400, requires_grad=True); y = net(x)
    y[0, 0, y.shape[-1]//2].backward(); nz = (x.grad[0, 0] != 0).nonzero()
    print(f"  brute force B={B}: receptive field = {nz.max().item()-nz.min().item()+1}")
lam = 1.5406; a = 4.0
for hkl in [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2)]:
    d = a/math.sqrt(sum(h*h for h in hkl)); print(f"  {hkl}: d={d:.4f} A, 2theta={2*math.degrees(math.asin(lam/2/d)):.3f} deg")
for a_ in [3.5, 4.0, 5.0, 6.0]:
    t1 = 2*math.degrees(math.asin(lam/2/(a_/math.sqrt(3)))); t2 = 2*math.degrees(math.asin(lam/2/(a_/2)))
    print(f"  a={a_}: 2theta(111)={t1:.2f}, 2theta(200)={t2:.2f}, gap={t2-t1:.2f} deg")
print("=== 7.2.3")
sig = lambda b: 1/(1+math.exp(-b))
print("  0.6^50 =", 0.6**50)
for b in [0, 1, 3]: print(f"  b_f={b}: f={sig(b):.4f}, f^50={sig(b)**50:.3e}, f^100={sig(b)**100:.3e}, f^199={sig(b)**199:.3e}")
fmin = math.exp(-0.01); print("  f_min=", fmin, " b_f >=", math.log(fmin/(1-fmin)))
# Lab 7.7 Part A setup: inspect actual forget gates
T, H = 200, 32
for fb in [0.0, 3.0]:
    stats = []
    for seed in range(10):
        torch.manual_seed(seed); net = torch.nn.LSTM(1, H, batch_first=True)
        with torch.no_grad():
            for name, p in net.named_parameters():
                if "bias" in name: p[H:2*H].fill_(fb/2)
        x = torch.randn(1, T, 1)
        Wih, Whh = net.weight_ih_l0, net.weight_hh_l0; b = net.bias_ih_l0+net.bias_hh_l0
        h = torch.zeros(H); c = torch.zeros(H); fs = []; os_ = []
        with torch.no_grad():
            for t in range(T):
                gts = Wih@x[0, t]+Whh@h+b; i_, f_, g_, o_ = gts.chunk(4)
                i_, f_, g_, o_ = torch.sigmoid(i_), torch.sigmoid(f_), torch.tanh(g_), torch.sigmoid(o_)
                c = f_*c+i_*g_; h = o_*torch.tanh(c); fs.append(f_.clone()); os_.append(o_.clone())
        F = torch.stack(fs)               # (T,H)
        logprod = torch.log(F[1:]).sum(0)  # product of f over steps 2..T per unit (lag ~199)
        stats.append((F.mean().item(), F.min().item(), F.max().item(), logprod.max().item(), (torch.log(F[-50:]).sum(0)).max().item(), torch.stack(os_).mean().item()))
    s = np.array(stats)
    print(f"  LSTM forget bias {fb}: mean f={s[:,0].mean():.3f}, range [{s[:,1].min():.3f}, {s[:,2].max():.3f}], "
          f"best-unit prod f over last 50 steps = {np.exp(s[:,4]).mean():.2e} (mean over seeds), over 199 steps = {np.exp(s[:,3]).mean():.2e}; mean o={s[:,5].mean():.3f}")
lab = {"RNN": [1.1e-1, 1.2e-3, 3.3e-7, 4.8e-12, 1.5e-21, 6.0e-40], "LSTM0": [5.5e-2, 7.9e-4, 7.1e-7, 1.6e-11, 1.5e-20, 2.0e-38], "LSTM3": [4.3e-2, 4.1e-2, 4.3e-2, 4.5e-2, 3.2e-2, 3.0e-1]}
lags = [1, 10, 25, 50, 100, 199]
for kname, v in lab.items():
    print(f"  {kname}: per-step factor lag 25->50 = {(v[3]/v[2])**(1/25):.3f}, 50->100 = {(v[4]/v[3])**(1/50):.3f}")
