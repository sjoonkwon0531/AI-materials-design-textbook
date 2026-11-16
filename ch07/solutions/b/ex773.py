"""Ex 7.7.3: equivariance error vs rotation angle about n=(1,1,1)/sqrt3, 200 test frames (2200-2399)."""
import os, importlib.util, numpy as np, torch
torch.set_num_threads(1)
L = "../../../ch07/labs/"
_s = importlib.util.spec_from_file_location("m77", L + "77_models.py"); M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)
R3, E3, F3, Z = M.load(300); R = R3[2200:2400]
def rot(theta, n):
    n = n / np.linalg.norm(n); K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
def err(model, Q, dtype):
    Rd = R.to(dtype); Qt = torch.as_tensor(Q, dtype=dtype)
    E0, F0 = M.energy_forces(model, Rd); E1, F1 = M.energy_forces(model, Rd @ Qt.T)
    return ((F1 - F0 @ Qt.T).abs().mean() / F0.abs().mean()).item(), (E1 - E0).abs().mean().item() * 1e3
angles = list(range(0, 181, 15)) + [1, 2, 5]
angles = sorted(angles)
n = np.ones(3)
for key in ("mlp_aug_N1000", "mlp_N1000", "schnet_N1000"):
    w = M.Wrapped(key.split("_N")[0], Z); w.load_state_dict(torch.load(L + f"77_{key}.pt")); w.eval()
    rows = []
    for dt in (torch.float32, torch.float64):
        w = w.to(dt)
        rows.append([err(w, rot(np.radians(a), n), dt) for a in angles])
    print(key)
    for a, r32, r64 in zip(angles, rows[0], rows[1]):
        print(f"   theta {a:3d} deg: rel F err f32 {r32[0]:.2e} (f64 {r64[0]:.2e}) | mean|dE| f32 {r32[1]:.2e} meV (f64 {r64[1]:.2e})")
    r = np.array([x[0] for x in rows[1]]); a = np.radians(np.array(angles))
    sel = (np.array(angles) > 0) & (np.array(angles) <= 15)
    print("   small-angle slope d(err)/d(theta) [per rad], f64:", np.round(r[sel] / a[sel], 3))
# float32 eps scale
print("float32 eps", torch.finfo(torch.float32).eps)
