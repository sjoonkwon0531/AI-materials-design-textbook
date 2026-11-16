"""Lab 7.3.2 (key 73_group): group theory with numbers.
Part A  generate point groups from generators (closure), check axioms, orders (C3v, D4h, Oh)
Part B  SO(2) and O(3): homomorphism check, det = +/-1, non-commutativity of 3D rotations
Part C  characters: decompose the vector rep and the H-permutation rep of NH3 (C3v) into irreps
Part D  E(3) acting on a molecule: composition rule of (R, t) and inverse
Runtime: < 2 s."""
import numpy as np
rng = np.random.default_rng(0)


def rot(axis, ang):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def mirror(n):
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    return np.eye(3) - 2 * np.outer(n, n)


def close(gens, tol=1e-9):
    """Smallest set of matrices containing the generators and closed under multiplication."""
    G = [np.eye(3)]
    frontier = [np.eye(3)]
    while frontier:
        new = []
        for A in frontier:
            for g in gens:
                B = g @ A
                if not any(np.abs(B - C).max() < tol for C in G):
                    G.append(B); new.append(B)
        frontier = new
    return G


def index(G, M, tol=1e-9):
    return next(k for k, C in enumerate(G) if np.abs(C - M).max() < tol)


print("=== Part A: point groups from generators")
groups = {
    "C3v (NH3)": [rot([0, 0, 1], 2 * np.pi / 3), mirror([0, 1, 0])],
    "D4h (square planar)": [rot([0, 0, 1], np.pi / 2), rot([1, 0, 0], np.pi), -np.eye(3)],
    "Oh (cube, SF6)": [rot([0, 0, 1], np.pi / 2), rot([1, 1, 1], 2 * np.pi / 3), -np.eye(3)],
    "Td (CH4)": [rot([1, 1, 1], 2 * np.pi / 3), mirror([1, -1, 0]), rot([0, 0, 1], np.pi)],
}
for name, gens in groups.items():
    G = close(gens)
    n = len(G)
    closure = all(any(np.abs(A @ B - C).max() < 1e-9 for C in G) for A in G for B in G)
    inverses = all(any(np.abs(A @ B - np.eye(3)).max() < 1e-9 for B in G) for A in G)
    assoc = max(np.abs((A @ B) @ C - A @ (B @ C)).max() for A, B, C in
                [(G[i], G[j], G[k]) for i, j, k in rng.integers(0, n, (200, 3))])
    abel = all(np.abs(A @ B - B @ A).max() < 1e-9 for A in G for B in G)
    dets = {}
    for A in G:
        k = int(round(np.linalg.det(A))); dets[k] = dets.get(k, 0) + 1
    print(f"{name:26s} order {n:2d} | closure {closure} | inverses {inverses} | assoc. err {assoc:.1e} |"
          f" abelian {abel} | det(+1):{dets.get(1,0)} det(-1):{dets.get(-1,0)}")

print("=== Part B: continuous groups")
th = rng.uniform(0, 2 * np.pi, (1000, 2))
R2 = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
err = max(np.abs(R2(a) @ R2(b) - R2(a + b)).max() for a, b in th)
print(f"SO(2): max |D(a)D(b) - D(a+b)| over 1000 pairs = {err:.1e}  (abelian: D(a)D(b) = D(b)D(a))")
Rx, Rz = rot([1, 0, 0], np.pi / 2), rot([0, 0, 1], np.pi / 2)
print(f"SO(3): |Rx(90)Rz(90) - Rz(90)Rx(90)|_max = {np.abs(Rx @ Rz - Rz @ Rx).max():.3f}  -> non-abelian")
Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
print(f"random orthogonal Q: |Q^T Q - I| = {np.abs(Q.T @ Q - np.eye(3)).max():.1e}, det = {np.linalg.det(Q):+.6f}")

print("=== Part C: characters of C3v (NH3: N on the z-axis, H in the xz-mirror and its 120-degree images)")
G = close(groups["C3v (NH3)"])
H = np.array([[np.cos(a), np.sin(a), -0.35] for a in (0, 2 * np.pi / 3, 4 * np.pi / 3)])
H[:, :2] *= 0.94
# classes: identity, the two C3 rotations, the three mirrors
def cls(A):
    if np.abs(A - np.eye(3)).max() < 1e-9: return "E"
    return "2C3" if np.linalg.det(A) > 0 else "3sv"
classes = ["E", "2C3", "3sv"]; sizes = {"E": 1, "2C3": 2, "3sv": 3}
irreps = {"A1": {"E": 1, "2C3": 1, "3sv": 1}, "A2": {"E": 1, "2C3": 1, "3sv": -1}, "E": {"E": 2, "2C3": -1, "3sv": 0}}
def perm_matrix(A):
    P = np.zeros((3, 3))
    for i in range(3):
        j = int(np.argmin(np.linalg.norm(H - H[i] @ A.T, axis=1))); P[j, i] = 1
    return P
reps = {"vector (x,y,z)": lambda A: A, "H permutations": perm_matrix}
for rname, f in reps.items():
    chi = {}
    for A in G:
        chi.setdefault(cls(A), []).append(np.trace(f(A)))
    chi = {c: float(np.mean(v)) for c, v in chi.items()}
    mult = {ir: sum(sizes[c] * chi[c] * irreps[ir][c] for c in classes) / len(G) for ir in irreps}
    hom = max(np.abs(f(A @ B) - f(A) @ f(B)).max() for A in G for B in G)
    print(f"{rname:16s}: chi = {[round(chi[c], 6) + 0.0 for c in classes]} -> multiplicities "
          + ", ".join(f"{k}:{v:.0f}" for k, v in mult.items()) + f" | homomorphism err {hom:.1e}")
orth = {(a, b): sum(sizes[c] * irreps[a][c] * irreps[b][c] for c in classes) / len(G) for a in irreps for b in irreps}
print("character orthogonality <chi_a, chi_b>:", {f"{a}{b}": v for (a, b), v in orth.items() if a <= b})

print("=== Part D: E(3) elements g = (R, t) acting as x -> R x + t")
def act(g, X): return X @ g[0].T + g[1]
def compose(g, h): return (g[0] @ h[0], g[0] @ h[1] + g[1])
def inv(g): return (g[0].T, -g[0].T @ g[1])
X = rng.standard_normal((9, 3))
g = (rot(rng.standard_normal(3), 1.0) @ mirror([0, 0, 1]), rng.standard_normal(3))
h = (rot(rng.standard_normal(3), 2.0), rng.standard_normal(3))
print(f"|g.(h.X) - (gh).X| = {np.abs(act(g, act(h, X)) - act(compose(g, h), X)).max():.1e},"
      f"  |g^-1.(g.X) - X| = {np.abs(act(inv(g), act(g, X)) - X).max():.1e}")
gh, hg = compose(g, h), compose(h, g)
print(f"translations do not commute with rotations: |(gh).t - (hg).t| = {np.abs(gh[1] - hg[1]).max():.3f}")
Dg = np.linalg.norm(act(g, X)[:, None] - act(g, X)[None], axis=-1); D0 = np.linalg.norm(X[:, None] - X[None], axis=-1)
print(f"all pair distances preserved by g (det R = {np.linalg.det(g[0]):+.0f}): max change {np.abs(Dg - D0).max():.1e}")
