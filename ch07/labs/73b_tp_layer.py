"""Lab 73b_tp: Clebsch-Gordan tensor product and an equivariant convolution layer in e3nn.
(A) 1 (x) 1 = 0 + 1 + 2 via o3.wigner_3j: the three outputs are the dot product, cross product and
    traceless symmetric part; CG intertwines D1 (x) D1 with D_L.
(B) a TFN/NequIP-style point convolution: message_ij = TP(h_j, Y(r_ij); w(|r_ij|)), summed over j, then a Gate.
    Equivariance error under random O(3) elements, float32 and float64, on an ethanol geometry; translation, permutation.
(C) naive component-wise SiLU on l=1 features breaks equivariance; the Gate does not.
(D) parameter counts: equivariant layer vs an unconstrained linear map with the same in/out dimensions.
Runtime < 10 s on one CPU thread."""
import numpy as np, torch, time
from e3nn import o3, nn as enn
from e3nn.math import soft_one_hot_linspace
torch.set_num_threads(1)
t0 = time.time()
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

print("=== Part A: 1 (x) 1 = 0 + 1 + 2 ===")
a, b = torch.randn(3), torch.randn(3)
# e3nn's wigner_3j has unit Frobenius norm; U_L = sqrt(2L+1) C_L makes the 9x9 change of basis orthogonal
out = {L: (2*L+1)**0.5 * torch.einsum("ijk,i,j->k", o3.wigner_3j(1, 1, L), a, b) for L in (0, 1, 2)}
print(f"  L=0: U0.(a,b) = {out[0].item():+.6f};  a.b/sqrt(3) = {(a @ b / 3**0.5).item():+.6f}")
cr = torch.linalg.cross(a, b) / 2**0.5
print(f"  L=1: max|U1.(a,b) - (a x b)/sqrt(2)| = {(out[1] - cr).abs().max():.1e}")
S = 0.5 * (torch.outer(a, b) + torch.outer(b, a)) - (a @ b) / 3 * torch.eye(3)
print(f"  L=2: |U2.(a,b)| = {out[2].norm():.6f};  ||traceless symmetric part of a b^T||_F = {S.norm():.6f}")
print(f"  dims: 3 x 3 = 9 = 1 + 3 + 5;  |a|^2|b|^2 = {(a.norm()*b.norm())**2:.6f} = sum_L |U_L.(a,b)|^2 = {sum(o.norm()**2 for o in out.values()):.6f}")
R = o3.rand_matrix()
for L in (0, 1, 2):
    C = o3.wigner_3j(1, 1, L)
    D1 = o3.Irrep(1, -1).D_from_matrix(R); DL = o3.Irrep(L, (-1)**L).D_from_matrix(R)
    lhs = torch.einsum("ijk,ia,jb->abk", C, D1, D1)        # C applied after rotating both inputs
    rhs = torch.einsum("abk,lk->abl", C, DL)               # D_L applied after C
    print(f"  intertwiner check L={L}: max|C(D1 x D1) - D_L C| = {(lhs - rhs).abs().max():.1e}")

print("=== Part B: equivariant point convolution + Gate on an ethanol geometry ===")
irreps_in = o3.Irreps("8x0e + 8x1o")            # node features
irreps_sh = o3.Irreps.spherical_harmonics(2)     # 1x0e + 1x1o + 1x2e
irreps_scalars, irreps_gated = o3.Irreps("8x0e"), o3.Irreps("8x1o + 8x2e")
irreps_gates = o3.Irreps(f"{irreps_gated.num_irreps}x0e")
irreps_mid = irreps_scalars + irreps_gates + irreps_gated

class PointConv(torch.nn.Module):
    """h_i' = Gate( sum_j TP(h_j, Y(r_ij); w(|r_ij|)) ), TFN/NequIP style."""
    def __init__(self, n_basis=8, r_cut=4.0):
        super().__init__()
        self.n_basis, self.r_cut = n_basis, r_cut
        self.tp = o3.FullyConnectedTensorProduct(irreps_in, irreps_sh, irreps_mid, shared_weights=False)
        self.radial = torch.nn.Sequential(torch.nn.Linear(n_basis, 16), torch.nn.SiLU(),
                                          torch.nn.Linear(16, self.tp.weight_numel))
        self.gate = enn.Gate(irreps_scalars, [torch.nn.functional.silu], irreps_gates, [torch.sigmoid], irreps_gated)
    def forward(self, h, pos):
        n = pos.shape[0]
        src, dst = torch.where(~torch.eye(n, dtype=bool))
        vec = pos[src] - pos[dst]
        dist = vec.norm(dim=1)
        emb = soft_one_hot_linspace(dist, 0.0, self.r_cut, self.n_basis, basis="smooth_finite", cutoff=True)
        Y = o3.spherical_harmonics(irreps_sh, vec, normalize=True, normalization="component")
        msg = self.tp(h[src], Y, self.radial(emb))
        agg = torch.zeros(n, irreps_mid.dim).index_add_(0, dst, msg)
        return self.gate(agg)

d = np.load("../ch07/data/ethanol_xtb_300K.npz")
for dtype in (torch.float32, torch.float64):
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)
    layer = PointConv()
    pos = torch.tensor(d["R"][0], dtype=dtype)
    h = irreps_in.randn(9, -1)
    out = layer(h, pos)
    irreps_out = layer.gate.irreps_out
    err_rot, err_inv = 0.0, 0.0
    for k in range(20):
        R = o3.rand_matrix()
        for Q, tag in ((R, "rot"), (-R, "inv")):
            o2 = layer(h @ irreps_in.D_from_matrix(Q).T, pos @ Q.T)
            e = (o2 - out @ irreps_out.D_from_matrix(Q).T).abs().max().item()
            if tag == "rot": err_rot = max(err_rot, e)
            else: err_inv = max(err_inv, e)
    t = torch.randn(3)
    err_tr = (layer(h, pos + t) - out).abs().max().item()
    perm = torch.randperm(9)
    err_pm = (layer(h[perm], pos[perm]) - out[perm]).abs().max().item()
    print(f"  {str(dtype)[6:]}: out irreps '{irreps_out}' (dim {irreps_out.dim}), |out|_max = {out.abs().max():.3f}")
    print(f"     max equivariance error: SO(3) {err_rot:.1e}, improper O(3) {err_inv:.1e}, translation {err_tr:.1e}, permutation {err_pm:.1e}")
torch.set_default_dtype(torch.float64)

print("=== Part C: which nonlinearity keeps equivariance? (l=1 block, 1000 random rotations) ===")
torch.manual_seed(1)
v = torch.randn(8, 3)
e_naive, e_gate, e_norm = 0.0, 0.0, 0.0
for k in range(1000):
    R = o3.rand_matrix()
    naive = lambda x: torch.nn.functional.silu(x)
    normgate = lambda x: torch.sigmoid(x.norm(dim=1, keepdim=True)) * x
    e_naive = max(e_naive, (naive(v @ R.T) - naive(v) @ R.T).abs().max().item())
    e_norm = max(e_norm, (normgate(v @ R.T) - normgate(v) @ R.T).abs().max().item())
print(f"  component-wise SiLU on vectors: max error = {e_naive:.2f}  (|v|_max = {v.abs().max():.2f})")
print(f"  norm gate sigma(|v|) v:         max error = {e_norm:.1e}")

print("=== Part D: parameter counts ===")
layer = PointConv()
n_tp = layer.tp.weight_numel; n_rad = sum(p.numel() for p in layer.radial.parameters())
print(f"  TP paths = {len(layer.tp.instructions)}, TP weights per edge (from radial MLP) = {n_tp}, radial MLP params = {n_rad}")
din, dsh, dout = irreps_in.dim, irreps_sh.dim, irreps_mid.dim
print(f"  an unconstrained bilinear map (in {din}) x (sh {dsh}) -> {dout} has {din*dsh*dout} entries "
      f"= {din*dsh*dout/n_tp:.1f} x the TP weights")
lin_e = o3.Linear(o3.Irreps("16x0e+16x1o+16x2e"), o3.Irreps("16x0e+16x1o+16x2e"))
d16 = o3.Irreps("16x0e+16x1o+16x2e").dim
print(f"  o3.Linear on 16x0e+16x1o+16x2e (dim {d16}): {lin_e.weight.numel()} weights (+0 bias for l>0); "
      f"torch.nn.Linear {d16}->{d16}: {d16*d16 + d16} params ({(d16*d16+d16)/lin_e.weight.numel():.1f}x)")
print(f"runtime {time.time()-t0:.1f} s")
