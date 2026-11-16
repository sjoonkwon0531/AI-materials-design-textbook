import math, numpy as np, torch, importlib.util, sys
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
print("=== 7.2.2 brute-force receptive field with avg pool (same geometry as max pool)")
for B in [3, 5]:
    layers = []
    for b in range(B): layers += [torch.nn.Conv1d(1, 1, 7, padding=3, bias=False), torch.nn.AvgPool1d(2)]
    net = torch.nn.Sequential(*layers)
    for l in net:
        if isinstance(l, torch.nn.Conv1d): torch.nn.init.constant_(l.weight, 1.0)
    x = torch.ones(1, 1, 1400, requires_grad=True); y = net(x); y[0, 0, y.shape[-1]//2].backward()
    nz = (x.grad[0, 0] != 0).nonzero(); print(f"  B={B}: receptive field = {nz.max().item()-nz.min().item()+1} channels")

print("=== 7.2.4 fcc Cu graphs")
from pymatgen.core import Structure, Lattice
a = 3.61
conv = Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), ["Cu"], [[0, 0, 0]]); prim = conv.get_primitive_structure()
print(f"  NN = a/sqrt2 = {a/math.sqrt(2):.4f} A, 2nd = {a:.4f} A; prim natoms {len(prim)}, conv natoms {len(conv)}")
for name, s in [("primitive", prim), ("conventional", conv)]:
    for rc in [3.0, 4.0]:
        ci, ni, img, d = s.get_neighbor_list(rc)
        pairs = {}
        for i, j in zip(ci, ni): pairs[(i, j)] = pairs.get((i, j), 0)+1
        selfimg = int((ci == ni).sum())
        print(f"  {name:12s} rc={rc}: directed edges {len(ci):3d}, per atom {len(ci)//len(s)}, distinct (i,j) pairs {len(pairs)},"
              f" max parallel {max(pairs.values())}, self-image edges {selfimg}, distances {sorted(set(np.round(d, 3)))}")
# bcc with same NN distance for the collapse argument
abcc = 2*2.5527/math.sqrt(3)
bcc = Structure.from_spacegroup("Im-3m", Lattice.cubic(abcc), ["Cu"], [[0, 0, 0]]).get_primitive_structure()
ci, ni, img, d = bcc.get_neighbor_list(2.7)
print(f"  bcc a={abcc:.4f} prim, rc=2.7: edges {len(ci)}, all self-image {bool((ci == ni).all())}, distances {sorted(set(np.round(d, 3)))}")

print("=== 7.2.5 mean vs sum aggregation across the cutoff")
rc = 5.0; fc = lambda r: 0.5*(np.cos(np.pi*r/rc)+1)*(r < rc)
others = np.array([2.0, 2.5, 3.0, 3.5]); M = 1.0
for rj in [5.01, 5.0001, 4.9999, 4.99]:
    rs = np.append(others, rj); inside = rs < rc
    s_sum = (M*fc(rs)).sum(); s_mean = (M*fc(rs))[inside].sum()/inside.sum()
    s_wmean = (M*fc(rs)).sum()/fc(rs).sum()
    print(f"  r_j={rj}: sum={s_sum:.6f}  mean={s_mean:.6f}  fc-weighted mean={s_wmean:.6f}")

print("=== 7.2.6 SchNet layer with hand-set weights vs ACSF G2")
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
U = load("u", "../ch07/labs/72b_utils.py"); sn = load("sn", "../ch07/labs/72b_schnet.py")
D = np.load("../ch07/data/ethanol_xtb_300K.npz"); print("  keys", list(D.keys()))
Z = D["Z"] if "Z" in D else D["numbers"]; Rall = D["R"] if "R" in D else D["positions"]
Zt = torch.tensor(Z); frames = [0, 500, 1000, 1500, 2999]
R = torch.tensor(Rall[frames], dtype=torch.float64)
acsf = U.ACSF(np.asarray(Z)).double(); G = acsf(R)[..., :24]
species = acsf.species; rs = acsf.rs.double(); eta = 4.0
print("  species", species, "rs", np.round(rs.numpy(), 4))
# Interaction layer with n_feat = 24, n_rbf = 8
blk = sn.Interaction(24, 8, 5.0).double()
with torch.no_grad():
    # embedding in Derivation-Box form: one-hot (3) then W_in (24x3) replicating each element 8 times
    Win = torch.zeros(24, 3)
    for s in range(3): Win[s*8:(s+1)*8, s] = 1.0
    blk.in2f.weight.copy_(torch.eye(24))                  # in2f = identity (W_in folded into the embedding below)
    BIG = 30.0
    blk.filt[0].weight.copy_(torch.cat([torch.eye(8)]*3, 0)); blk.filt[0].bias.fill_(BIG)   # tile 8 Gaussians -> 24
    blk.filt[2].weight.copy_(torch.eye(24)); blk.filt[2].bias.fill_(-(BIG-math.log(2.0)))   # undo the shift of ssp
onehot = torch.tensor([[1.0 if z == s else 0.0 for s in species] for z in Z])
h0 = onehot @ Win.T                                         # (9,24)
n = len(Z); maxerr = 0.0
for b in range(len(frames)):
    ei = sn.full_graph_edges(n, 1); i, j = ei
    vec = R[b][j]-R[b][i]; r = vec.norm(dim=-1)
    rbf = torch.exp(-eta*(r[:, None]-rs[None, :])**2)      # Gaussian expansion with gamma = eta, mu = r_s
    W = blk.filt(rbf)*sn.cosine_cutoff(r, 5.0)[:, None]
    m = torch.zeros(n, 24).index_add_(0, i, blk.in2f(h0)[j]*W)
    maxerr = max(maxerr, (m-G[b]).abs().max().item())
print(f"  max |m_SchNet - G2_ACSF| over {len(frames)} frames x 9 atoms x 24 = {maxerr:.1e}; |G2|max = {G.abs().max().item():.3f}")
print("  ssp(30) - (30 - ln2) =", (sn.ssp(torch.tensor(30.0))-(30-math.log(2))).item())

print("=== 7.2.7 Legendre sums")
from scipy.special import eval_legendre, sph_harm_y
th = math.acos(-1/3)
tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])/math.sqrt(3)
sq = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
def S(u, l):
    c = u@u.T; return sum(eval_legendre(l, c[j, k]) for j in range(4) for k in range(4) if j != k)
def Sy(u, l):
    th_ = np.arccos(np.clip(u[:, 2], -1, 1)); ph = np.arctan2(u[:, 1], u[:, 0])
    tot = sum(abs(sph_harm_y(l, m, th_, ph).sum())**2 for m in range(-l, l+1))
    return 4*np.pi/(2*l+1)*tot - 4
for l in range(1, 7):
    print(f"  l={l}: tet S={S(tet, l):+.4f} (addition thm {Sy(tet, l):+.4f}, 12 P_l(-1/3)={12*eval_legendre(l, -1/3):+.4f})"
          f"   square S={S(sq, l):+.4f} (addition {Sy(sq, l):+.4f}, 8P_l(0)+4(-1)^l={8*eval_legendre(l, 0)+4*(-1)**l:+.4f})")
# per-H angle sums at C for DimeNet-type l<=1 vs l=2
for name, u in [("tet", tet), ("square", sq)]:
    c = u@u.T
    print(f"  {name}: per-H sum_k cos = {[round(sum(c[i, k] for k in range(4) if k != i), 4) for i in range(4)]},"
          f" per-H sum_k P2(cos) = {[round(sum(eval_legendre(2, c[i, k]) for k in range(4) if k != i), 4) for i in range(4)]}")
