import torch, math
from e3nn import o3
from e3nn.math import normalize2mom
torch.set_default_dtype(torch.float64); torch.manual_seed(0)
f = normalize2mom(torch.nn.functional.silu); print("normalize2mom(silu) constant:", f.cst if hasattr(f, "cst") else f)
rtp = o3.ReducedTensorProducts("ij=ji", i="1o"); C = rtp.change_of_basis.reshape(6, 9).T
def feats(r, q):
    mu = (q[:, None]*r).sum(0); Q = sum(q[j]*(torch.outer(r[j], r[j])-r[j].dot(r[j])/3*torch.eye(3)) for j in range(len(q)))
    c = torch.linalg.lstsq(C, Q.reshape(9, 1)).solution[:, 0]; return mu, c[1:], Q
tp = o3.FullyConnectedTensorProduct("1x1o", "1x2e", "1x1o")
r = torch.randn(6, 3); q = torch.randn(6); mu, q2, Q = feats(r, q); out = tp(mu[None], q2[None])[0]
R = o3.rand_matrix(); mu_r, q2_r, _ = feats(r@R.T, q)
print("mu' - D1 mu:", (mu_r - o3.Irrep("1o").D_from_matrix(R)@mu).abs().max().item(), " q2' - D2 q2:", (q2_r - o3.Irrep("2e").D_from_matrix(R)@q2).abs().max().item())
print("rotation test |f(Rr) - D1 f(r)|:", (tp(mu_r[None], q2_r[None])[0] - R@out).abs().max().item())
mu_i, q2_i, _ = feats(-r, q); print("inversion: q2(-r)-q2(r)", (q2_i-q2).abs().max().item(), " f(-r)+f(r):", (tp(mu_i[None], q2_i[None])[0] + out).abs().max().item())
