import torch, numpy as np
torch.set_num_threads(1)
T, H = 200, 32
def grad_vs_lag(seed, fbias, freeze_gates=False):
    torch.manual_seed(seed); net = torch.nn.LSTM(1, H, batch_first=True)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if "bias" in name: p[H:2*H].fill_(fbias/2)
    x = torch.randn(1, T, 1, requires_grad=True)
    out, _ = net(x)
    g, = torch.autograd.grad(out[0, -1].sum(), x)
    return g[0, :, 0].abs().numpy()[::-1]
lags = [1, 10, 25, 50, 100, 150, 199]
for fb in [3.0]:
    G = np.array([grad_vs_lag(s, fb) for s in range(10)])
    for l in lags: print(f"fb={fb} lag {l:3d}: per-seed " + " ".join(f"{v:.1e}" for v in G[:, l]) + f" | mean {G[:, l].mean():.1e} median {np.median(G[:, l]):.1e}")
# manual LSTM where gate pre-activations from h are removed (gates depend only on bias + x): isolates cell path
def manual(seed, fbias, cut_h_to_gates):
    torch.manual_seed(seed); net = torch.nn.LSTM(1, H, batch_first=True)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if "bias" in name: p[H:2*H].fill_(fbias/2)
    x = torch.randn(T, requires_grad=True)
    Wih, Whh = net.weight_ih_l0, net.weight_hh_l0; b = net.bias_ih_l0+net.bias_hh_l0
    h = torch.zeros(H); c = torch.zeros(H)
    for t in range(T):
        rec = (Whh@h) if not cut_h_to_gates else (Whh@h.detach())
        gts = Wih[:, 0]*x[t]+rec+b; i_, f_, g_, o_ = gts.chunk(4)
        c = torch.sigmoid(f_)*c+torch.sigmoid(i_)*torch.tanh(g_); h = torch.sigmoid(o_)*torch.tanh(c)
    gr, = torch.autograd.grad(h.sum(), x); return gr.abs().detach().numpy()[::-1]
for cut in [False, True]:
    G = np.array([manual(s, 3.0, cut) for s in range(10)])
    print(f"manual fb=3, recurrent h->gates path {'cut' if cut else 'kept'}: mean |dh_T/dx_t| at lags", {l: f"{G[:, l].mean():.1e}" for l in lags})
