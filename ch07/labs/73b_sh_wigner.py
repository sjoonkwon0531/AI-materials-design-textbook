"""Lab 73b_sh: real spherical harmonics, Wigner D and parity, checked numerically with e3nn.
(A) Y_l(R r) = D^l(R) Y_l(r) for l = 0..6 in float32 / float64
(B) D^l orthogonal; block-diagonal D for a direct sum of irreps
(C) parity: Y_l(-r) = (-1)^l Y_l(r); D of inversion for 1o vs 1e; cross product is 1e
(D) orthonormality of e3nn 'integral' normalization by Gauss-Legendre x uniform-phi quadrature
(E) link to Ch.6: SOAP-like coefficients c_l = sum_j f(r_j) Y_l(r_j) of an ethanol C atom transform with D^l,
    and their power spectrum |c_l|^2 is invariant.
Runtime < 10 s on one CPU thread."""
import numpy as np, torch, time
from e3nn import o3
torch.set_num_threads(1)
t0 = time.time()
torch.manual_seed(0)

print("=== Part A: Y_l(Rr) = D^l(R) Y_l(r), 1000 random unit vectors, 10 random rotations ===")
for dtype in (torch.float32, torch.float64):
    torch.set_default_dtype(dtype)
    g = torch.Generator().manual_seed(1)
    r = torch.randn(1000, 3, generator=g)
    errs = []
    for l in range(7):
        e = 0.0
        for k in range(10):
            R = o3.rand_matrix()
            Y = o3.spherical_harmonics(l, r, normalize=True, normalization="component")
            YR = o3.spherical_harmonics(l, r @ R.T, normalize=True, normalization="component")
            D = o3.Irrep(l, (-1) ** l).D_from_matrix(R)
            e = max(e, (YR - Y @ D.T).abs().max().item())
        errs.append(e)
    print(f"  {str(dtype)[6:]:8s} max error by l=0..6: " + " ".join(f"{x:.1e}" for x in errs))
torch.set_default_dtype(torch.float64)

print("=== Part B: D^l is orthogonal; a direct sum gives a block-diagonal D ===")
R = o3.rand_matrix()
for l in (1, 2, 3, 6):
    D = o3.Irrep(l, 1).D_from_matrix(R)
    print(f"  l={l}: shape {tuple(D.shape)}, ||D^T D - I||_max = {(D.T @ D - torch.eye(2*l+1)).abs().max():.1e}, det = {torch.det(D):+.6f}")
ir = o3.Irreps("2x0e + 1x1o + 1x2e")
D = ir.D_from_matrix(R)
blocks = [(0, 1), (1, 2), (2, 5), (5, 10)]
off = D.clone()
for a, b in blocks:
    off[a:b, a:b] = 0
print(f"  Irreps '{ir}' dim={ir.dim}; max |off-block entry| = {off.abs().max():.1e}")
print(f"  Dimension count: sum (2l+1) for l=0..6 = {sum(2*l+1 for l in range(7))} = (l_max+1)^2 = 49")

print("=== Part C: parity ===")
r = torch.randn(500, 3)
for l in range(5):
    Y = o3.spherical_harmonics(l, r, normalize=True)
    Ym = o3.spherical_harmonics(l, -r, normalize=True)
    print(f"  l={l}: max|Y(-r) - (-1)^l Y(r)| = {(Ym - (-1)**l * Y).abs().max():.1e}")
I = -torch.eye(3)
print("  D(inversion) for 1o:", o3.Irreps("1o").D_from_matrix(I).diagonal().tolist(),
      " for 1e:", o3.Irreps("1e").D_from_matrix(I).diagonal().tolist())
a, b = torch.randn(3), torch.randn(3)
for name, M in (("proper rotation", o3.rand_matrix()), ("improper (rotation x inversion)", -o3.rand_matrix())):
    lhs = torch.linalg.cross(M @ a, M @ b)
    rhs_vec = M @ torch.linalg.cross(a, b)          # how a 1o vector would transform
    rhs_pv = torch.det(M) * (M @ torch.linalg.cross(a, b))  # how a 1e pseudovector transforms
    print(f"  cross product under {name}: |err as 1o| = {(lhs-rhs_vec).abs().max():.1e}, |err as 1e| = {(lhs-rhs_pv).abs().max():.1e}")

print("=== Part D: orthonormality (e3nn 'integral' normalization), quadrature 40 x 80 ===")
x, w = np.polynomial.legendre.leggauss(40)
phi = np.arange(80) * 2 * np.pi / 80
ct, ph = np.meshgrid(x, phi, indexing="ij")
st = np.sqrt(1 - ct**2)
pts = torch.tensor(np.stack([st*np.cos(ph), st*np.sin(ph), ct], -1).reshape(-1, 3))
wt = torch.tensor((w[:, None] * np.full(80, 2*np.pi/80)[None, :]).reshape(-1))
Y = o3.spherical_harmonics(list(range(5)), pts, normalize=True, normalization="integral")  # 25 functions
G = (Y * wt[:, None]).T @ Y
print(f"  Gram matrix {tuple(G.shape)}: max|G - I| = {(G - torch.eye(25)).abs().max():.1e}")

print("=== Part E (link to Ch.6 Derivation Box 6.3): SOAP-like coefficients of ethanol atom C1 ===")
d = np.load("../../ch07/data/ethanol_xtb_300K.npz")
X = torch.tensor(d["R"][0])
rij = X - X[0]
rij = rij[1:]
dist = rij.norm(dim=1)
fc = 0.5 * (torch.cos(np.pi * dist / 4.0) + 1) * (dist < 4.0)
for l in range(4):
    R = o3.rand_matrix()
    c = (fc[:, None] * o3.spherical_harmonics(l, rij, normalize=True)).sum(0)
    cR = (fc[:, None] * o3.spherical_harmonics(l, rij @ R.T, normalize=True)).sum(0)
    D = o3.Irrep(l, (-1)**l).D_from_matrix(R)
    print(f"  l={l}: |c_l| = {c.norm():.6f}, ||c_l(R) - D c_l||_max = {(cR - D @ c).abs().max():.1e}, "
          f"| |c_l(R)|^2 - |c_l|^2 | = {abs(cR.norm()**2 - c.norm()**2):.1e}, ||c_l(R)-c_l||_max = {(cR-c).abs().max():.3f}")
print(f"runtime {time.time()-t0:.1f} s")
