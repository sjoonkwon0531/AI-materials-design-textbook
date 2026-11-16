"""Ex 1.6.G: Newton fractals for z^4-1, z^3-2z+2; relaxation a; boundary area vs max_iter."""
import numpy as np
def newton_grid(f, df, roots, a=1.0, N=600, span=2.0, max_iter=50, tol=1e-6):
    """Same rule as Lab 1.6 but iterates only the still-active points (faster)."""
    x = np.linspace(-span, span, N); X, Y = np.meshgrid(x, x)
    Z = (X + 1j*Y).ravel(); idx = np.arange(Z.size)
    basin = np.full(Z.size, -1); itc = np.zeros(Z.size, int)
    for k in range(max_iter):
        d = df(Z); safe = np.abs(d) > 1e-12
        st = np.zeros_like(Z); st[safe] = f(Z[safe])/d[safe]
        Z = Z - a*st
        done = np.zeros(Z.size, bool)
        for i, r in enumerate(roots):
            hit = (~done) & (np.abs(Z - r) < tol); basin[idx[hit]] = i; itc[idx[hit]] = k+1; done |= hit
        Z, idx = Z[~done], idx[~done]
        if Z.size == 0: break
    return basin.reshape(N, N), itc.reshape(N, N)
F = {"z^3-1":    (lambda z: z**3-1,      lambda z: 3*z**2,   np.roots([1,0,0,-1])),
     "z^4-1":    (lambda z: z**4-1,      lambda z: 4*z**3,   np.roots([1,0,0,0,-1])),
     "z^3-2z+2": (lambda z: z**3-2*z+2,  lambda z: 3*z**2-2, np.roots([1,0,-2,2]))}
print("(a)")
for name in ["z^4-1", "z^3-2z+2"]:
    f, df, r = F[name]; B, I = newton_grid(f, df, r)
    print(f"  {name}: roots={np.round(r,4)}, converged={np.mean(B>=0):.4f}, "
          f"mean iters={I[B>=0].mean():.3f}, basin fr={[round(np.mean(B==i),4) for i in range(len(r))]}")
# 2-cycle check for z^3-2z+2
f, df, r = F["z^3-2z+2"]; z = 0.0
seq = [z]
for _ in range(4): z = z - f(z)/df(z); seq.append(z)
print("  z^3-2z+2 orbit of 0:", seq)
N = lambda z: z - f(z)/df(z); h = 1e-6
dN = lambda z: (N(z+h)-N(z-h))/(2*h)
print(f"  multiplier of 2-cycle N'(0)N'(1) = {dN(0.0)*dN(1.0):.3e}")
print("(b) z^3-1 with relaxation a, max_iter=200")
f, df, r = F["z^3-1"]
for a in [0.5, 1.0, 1.5]:
    B, I = newton_grid(f, df, r, a=a, max_iter=200)
    print(f"  a={a}: converged={np.mean(B>=0):.4f}, mean iters={I[B>=0].mean():.3f}, max iters={I.max()}")
print("(c) non-converged fraction vs max_iter")
for name in ["z^3-1", "z^4-1", "z^3-2z+2"]:
    f, df, r = F[name]; out = []
    for mi in [10, 20, 40, 80, 160, 320]:
        B, I = newton_grid(f, df, r, max_iter=mi); out.append(f"{mi}:{np.mean(B<0):.5f}")
    print(f"  {name}: " + ", ".join(out))
print("(c') z^4-1 grid refinement at max_iter=320")
f, df, r = F["z^4-1"]
for N_ in [300, 600, 1200, 1201]:
    B, I = newton_grid(f, df, r, N=N_, max_iter=320); print(f"  N={N_}: {np.mean(B<0):.6f}")
print("(b') boundary-pixel density (pixels whose right/down neighbour has another basin), z^3-1, max_iter=200")
f, df, r = F["z^3-1"]
for a in [0.5, 1.0, 1.5]:
    B, I = newton_grid(f, df, r, a=a, max_iter=200)
    bd = np.zeros_like(B, bool); bd[:-1] |= B[:-1] != B[1:]; bd[:, :-1] |= B[:, :-1] != B[:, 1:]
    print(f"  a={a}: boundary pixel fraction={bd.mean():.4f}")
