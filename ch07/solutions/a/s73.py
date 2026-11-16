import math, numpy as np, torch, importlib.util, time, itertools
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
spec = importlib.util.spec_from_file_location("grp", "../ch07/labs/73a_group.py")
src = open("../ch07/labs/73a_group.py").read().split('print("=== Part A')[0]
grp = {}; exec(src, grp); rot, mirror, close = grp["rot"], grp["mirror"], grp["close"]
print("=== 7.3.1 S3 non-abelian")
def comp(p, q): return tuple(p[q[i]] for i in range(3))     # (p o q)(i) = p(q(i))
S3 = list(itertools.permutations(range(3))); print("  |S3| =", len(S3))
a, b = (1, 0, 2), (0, 2, 1); print("  (12)o(23) =", comp(a, b), " (23)o(12) =", comp(b, a))
Q1 = np.diag([-1., 1, 1]); Q2 = np.diag([1., -1, 1]); print("  det of product of two det=-1 matrices:", np.linalg.det(Q1@Q2))
print("=== 7.3.3 Td characters")
Td = close([rot([1, 1, 1], 2*np.pi/3), mirror([1, -1, 0]), rot([0, 0, 1], np.pi)])
H = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])/math.sqrt(3)*1.09
def cls(A):
    d = np.linalg.det(A); tr = np.trace(A)
    if d > 0:
        if abs(tr-3) < 1e-9: return "E"
        return "8C3" if abs(tr) < 1e-9 else "3C2"
    return "6S4" if abs(tr+1) < 1e-9 else "6sd"
classes = ["E", "8C3", "3C2", "6S4", "6sd"]; size = {c: 0 for c in classes}
chiv, chip = {}, {}
for A in Td:
    c = cls(A); size[c] += 1; chiv[c] = np.trace(A)
    perm = [int(np.argmin(np.linalg.norm(H - (A@h), axis=1))) for h in H]
    assert all(np.linalg.norm(H[perm[k]] - A@H[k]) < 1e-9 for k in range(4))
    chip[c] = sum(perm[k] == k for k in range(4))
print("  order", len(Td), "class sizes", size)
print("  vector chi", [round(chiv[c], 6) for c in classes], " perm chi", [chip[c] for c in classes])
irr = {"A1": [1, 1, 1, 1, 1], "A2": [1, 1, 1, -1, -1], "E": [2, -1, 2, 0, 0], "T1": [3, 0, -1, 1, -1], "T2": [3, 0, -1, -1, 1]}
for name, chi in [("vector", chiv), ("H perm", chip)]:
    mult = {a: sum(size[c]*chi[c]*irr[a][k] for k, c in enumerate(classes))/24 for a in irr}
    print(f"  {name}: ", {a: round(v, 6) for a, v in mult.items()})
# spherical harmonics restricted to Td: multiplicity of A1 in l (with parity (-1)^l)
from e3nn import o3
for l in range(7):
    D = [o3.Irrep(l, (-1)**l).D_from_matrix(torch.tensor(A)).numpy() for A in Td]
    print(f"  l={l}: multiplicity of A1 (Td-invariant harmonics) = {np.mean([np.trace(d) for d in D]):.3f}")
print("=== 7.3.5 tensor decomposition")
rng = np.random.default_rng(0); T = rng.standard_normal((3, 3))
def parts(T):
    T0 = np.trace(T)/3*np.eye(3); T1 = 0.5*(T-T.T); T2 = 0.5*(T+T.T)-T0; return T0, T1, T2
eps = np.zeros((3, 3, 3))
for i, j, k in itertools.permutations(range(3)): eps[i, j, k] = np.linalg.det(np.eye(3)[[i, j, k]])
w = lambda T: 0.5*np.einsum("kij,ij->k", eps, T)
T0, T1, T2 = parts(T)
Qs = [o3.rand_matrix().numpy() for _ in range(100)]
e_tr = max(abs(np.trace(Q@T@Q.T)-np.trace(T)) for Q in Qs)
e1 = max(abs(np.linalg.norm(parts(Q@T@Q.T)[1])-np.linalg.norm(T1)) for Q in Qs)
e2 = max(abs(np.linalg.norm(parts(Q@T@Q.T)[2])-np.linalg.norm(T2)) for Q in Qs)
emix = max(max(np.abs(parts(Q@T@Q.T)[k]-Q@parts(T)[k]@Q.T).max() for k in range(3)) for Q in Qs)
print(f"  max invariance errors over 100 Q: tr {e_tr:.1e}, ||T1|| {e1:.1e}, ||T2|| {e2:.1e}; piecewise (QTQ^T)^(k) - Q T^(k) Q^T: {emix:.1e}")
print(f"  tr T={np.trace(T):.6f}, ||T1||_F={np.linalg.norm(T1):.6f}, ||T2||_F={np.linalg.norm(T2):.6f}")
# axial vector
ew = max(np.abs(w(Q@T@Q.T) - np.linalg.det(Q)*Q@w(T)).max() for Q in Qs + [-Q for Q in Qs])
print(f"  axial: max |w(QTQ^T) - det(Q) Q w(T)| over 200 O(3) elements = {ew:.1e};  w under inversion: {w(T)} -> {w((-np.eye(3))@T@(-np.eye(3)))}")
# 5-dim rep
B = [np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])/math.sqrt(2), np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])/math.sqrt(6)]
for (i, j) in [(0, 1), (0, 2), (1, 2)]:
    M = np.zeros((3, 3)); M[i, j] = M[j, i] = 1/math.sqrt(2); B.append(M)
Gram = np.array([[np.sum(a*b) for b in B] for a in B]); print("  basis Gram - I max:", np.abs(Gram-np.eye(5)).max())
def M5(Q): return np.array([[np.sum(B[a]*(Q@B[b]@Q.T)) for b in range(5)] for a in range(5)])
eo = max(np.abs(M5(Q).T@M5(Q)-np.eye(5)).max() for Q in Qs); dets = [np.linalg.det(M5(Q)) for Q in Qs]
eh = max(np.abs(M5(Qs[k]@Qs[k+1])-M5(Qs[k])@M5(Qs[k+1])).max() for k in range(99))
def angle(Q): return math.acos(max(-1, min(1, (np.trace(Q)-1)/2)))
ech = max(abs(np.trace(M5(Q)) - (1+2*math.cos(angle(Q))+2*math.cos(2*angle(Q)))) for Q in Qs)
eD = max(abs(np.trace(M5(Q)) - np.trace(o3.Irrep(2, 1).D_from_matrix(torch.tensor(Q)).numpy())) for Q in Qs)
t2 = np.array([np.sum(b*T2) for b in B]); ecoef = max(np.abs(np.array([np.sum(b*(Q@T2@Q.T)) for b in B]) - M5(Q)@t2).max() for Q in Qs)
print(f"  M5: ||M^T M - I|| {eo:.1e}, det range [{min(dets):.6f},{max(dets):.6f}], homomorphism err {eh:.1e}, coefficient transform err {ecoef:.1e}")
print(f"  character tr M5 vs 1+2cos(phi)+2cos(2phi): {ech:.1e};  vs e3nn D^2 trace: {eD:.1e}")
rtp = o3.ReducedTensorProducts("ij=ji", i="1o"); print("  e3nn ReducedTensorProducts ij=ji (1o):", rtp.irreps_out)
rtp2 = o3.ReducedTensorProducts("ij", i="1o", j="1o"); print("  e3nn ReducedTensorProducts ij (1o):", rtp2.irreps_out)
# symmetric output: assemble from 0e+2e
cob = rtp.change_of_basis   # (6? dim_out, 3, 3)
x = torch.randn(rtp.irreps_out.dim); Tm = torch.einsum("k,kij->ij", x, cob)
Q = o3.rand_matrix(); xr = rtp.irreps_out.D_from_matrix(Q)@x; Tr = torch.einsum("k,kij->ij", xr, cob)
print(f"  assembled from 0e+2e: symmetric err {(Tm-Tm.T).abs().max():.1e}; |T(Dx) - Q T(x) Q^T| = {(Tr - Q@Tm@Q.T).abs().max():.1e}")
print("=== 7.3.6 decompositions")
for a_, b_ in [("2e", "2e"), ("1o", "3o"), ("2e", "3o"), ("1o", "2e"), ("1o", "1e"), ("1o", "1o")]:
    print(f"  {a_} x {b_} =", (o3.Irrep(a_)*o3.Irrep(b_)) if False else o3.Irreps(a_).__mul__ if False else [str(ir) for ir in o3.Irrep(a_)*o3.Irrep(b_)])
r3 = o3.ReducedTensorProducts("ijk", i="1o", j="1o", k="1o"); print("  ijk:", r3.irreps_out, "dim", r3.irreps_out.dim)
t0 = time.time(); r4 = o3.ReducedTensorProducts("ijkl=jikl=ijlk=klij", i="1o"); print("  Cijkl:", r4.irreps_out, "dim", r4.irreps_out.dim, f"({time.time()-t0:.1f} s)")
print("=== 7.3.7")
Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]]); relu = lambda v: np.maximum(v, 0)
for v in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (1, 1, 0), (0.3, -0.7, 0.2)]:
    v = np.array(v, float); print(f"  v={v}: ReLU(Rv)={relu(Rz@v)}, R ReLU(v)={Rz@relu(v)+0.0}, equal {np.allclose(relu(Rz@v), Rz@relu(v))}")
rng = np.random.default_rng(1); V = rng.standard_normal((1000, 3)); Rs = [o3.rand_matrix().numpy() for _ in range(100)]
f1 = lambda v: np.tanh(np.linalg.norm(v, axis=-1, keepdims=True))*v; f2 = lambda v: v*v[..., 2:3]
e1 = max(np.abs(f1(V@R.T) - f1(V)@R.T).max() for R in Rs); e2 = max(np.abs(f2(V@R.T) - f2(V)@R.T).max() for R in Rs)
Rzs = [np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]]) for t in rng.uniform(0, 6.28, 100)]
e2z = max(np.abs(f2(V@R.T) - f2(V)@R.T).max() for R in Rzs)
print(f"  tanh(|v|)v: max err {e1:.1e};  v(v.e_z): max err {e2:.2f} (|v|max {np.linalg.norm(V,axis=1).max():.2f}); for rotations about z only: {e2z:.1e}")
print("=== 7.3.8 1o x 2e -> 1o")
r = torch.randn(6, 3); q = torch.randn(6)
mu = (q[:, None]*r).sum(0)
Qt = sum(q[j]*(torch.outer(r[j], r[j]) - r[j].dot(r[j])/3*torch.eye(3)) for j in range(6))
print("  ", [str(ir) for ir in o3.Irrep("1o")*o3.Irrep("2e")])
cobs = rtp.change_of_basis                     # maps 0e+2e coefficients -> symmetric 3x3
# coefficients of Q in the 2e part: solve least squares
Cmat = cobs.reshape(cobs.shape[0], 9).T        # 9 x 6
coef = torch.linalg.lstsq(Cmat, Qt.reshape(9, 1)).solution[:, 0]; q2 = coef[1:]   # drop 0e (zero)
print(f"  0e coefficient of traceless Q = {coef[0].item():.1e}; reconstruction err {(Cmat@coef - Qt.reshape(9)).abs().max():.1e}")
tp = o3.FullyConnectedTensorProduct("1x1o", "1x2e", "1x1o"); print("  weight_numel", tp.weight_numel, "paths", len(tp.instructions))
out = tp(mu[None], q2[None])[0]; Qmu = Qt@mu
print(f"  tp output / (Q mu) componentwise: {(out/Qmu).tolist()}")
ratios = []
for s in range(5):
    r = torch.randn(6, 3); q = torch.randn(6); mu = (q[:, None]*r).sum(0)
    Qt = sum(q[j]*(torch.outer(r[j], r[j]) - r[j].dot(r[j])/3*torch.eye(3)) for j in range(6))
    coef = torch.linalg.lstsq(Cmat, Qt.reshape(9, 1)).solution[:, 0]; o = tp(mu[None], coef[None, 1:])[0]
    ratios.append((o/(Qt@mu)).tolist())
print("  ratio over 5 random inputs (should be one constant c):", np.round(np.array(ratios), 6).ravel()[:6], "... spread", np.ptp(np.array(ratios)))
R = o3.rand_matrix(); D1 = o3.Irrep("1o").D_from_matrix(R); D2 = o3.Irrep("2e").D_from_matrix(R)
e_eq = (tp((D1@mu)[None], (D2@coef[1:])[None])[0] - D1@tp(mu[None], coef[None, 1:])[0]).abs().max()
r_rot = r@R.T; mu_r = (q[:, None]*r_rot).sum(0); Qr = sum(q[j]*(torch.outer(r_rot[j], r_rot[j]) - r_rot[j].dot(r_rot[j])/3*torch.eye(3)) for j in range(6))
print(f"  D1 = R? {(D1-R).abs().max():.1e};  equivariance |tp(D1 mu, D2 q2) - D1 tp| = {e_eq:.1e};  (RQR^T)(R mu) - R(Q mu) = {(Qr@mu_r - R@(Qt@mu)).abs().max():.1e}")
print("=== 7.3.9 parameter counts")
ins = o3.Irreps("16x0e+16x1o+16x2e"); sh = o3.Irreps("0e+1o+2e")
for outs in ["16x0e+16x1o+16x2e", "16x0e+16x1o+16x2e+16x1e", "16x0e+16x1o+16x2e+16x2o", "16x0e+16x1o+16x2e+16x1e+16x2o"]:
    tp = o3.FullyConnectedTensorProduct(ins, sh, o3.Irreps(outs), shared_weights=False)
    paths = [(str(ins[i.i_in1].ir), str(sh[i.i_in2].ir), str(o3.Irreps(outs)[i.i_out].ir)) for i in tp.instructions]
    print(f"  out {outs}: paths {len(paths)}, weights/edge {tp.weight_numel}")
    if outs == "16x0e+16x1o+16x2e": print("   ", paths)
print("  unconstrained bilinear 144x9x144 =", 144*9*144, " ratio", 144*9*144/2816)
print("=== 7.3.10")
n, nv = 12, 64; rng = np.random.default_rng(0); phi = rng.standard_normal((n, nv))
t0 = time.time(); A = phi.sum(0); P1 = np.einsum("a,b,c->abc", A, A, A); t1 = time.time()-t0
t0 = time.time(); P2 = np.einsum("ja,kb,lc->abc", phi, phi, phi, optimize=False); t2 = time.time()-t0
print(f"  direct triple sum vs product of A: max rel diff {np.abs(P1-P2).max()/np.abs(P1).max():.1e}; time {t2:.2f} s vs {t1*1e3:.1f} ms")
print("  mult counts: direct", 2*nv**3*n**3, " via A", 2*nv**3, " ratio", n**3)
# self-interaction split for nu=2
P = np.outer(A, A); off = sum(np.outer(phi[j], phi[k]) for j in range(n) for k in range(n) if j != k); diag = sum(np.outer(phi[j], phi[j]) for j in range(n))
print("  A A = offdiag + diag check:", np.abs(P - off - diag).max())
for name, u in [("tet", np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])/math.sqrt(3)), ("square", np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0.]]))]:
    norms = [o3.spherical_harmonics(l, torch.tensor(u), normalize=True).sum(0).norm().item() for l in range(7)]
    print(f"  {name}: |A_l| (unit weights) l=0..6:", [f"{x:.4f}" for x in norms])
