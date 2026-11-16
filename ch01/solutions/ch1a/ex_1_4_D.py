# 연습문제 1.4.D: Hilbert 계 H_n x = b (b=행합, 정답 x=1)를 LU/Cholesky/QR/CG로
import numpy as np, scipy.linalg as sl, warnings
from scipy.sparse.linalg import cg
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
def solve_all(n):
    H = sl.hilbert(n); b = H.sum(1); xe = np.ones(n); out = {}
    out["LU"] = sl.lu_solve(sl.lu_factor(H), b)
    try: out["Cholesky"] = sl.cho_solve(sl.cho_factor(H), b)
    except np.linalg.LinAlgError: out["Cholesky"] = None
    Q, Rm = np.linalg.qr(H); out["QR"] = sl.solve_triangular(Rm, Q.T @ b)
    it = {"k": 0}
    def cb(xk): it["k"] += 1
    x, info = cg(H, b, rtol=1e-15, atol=0, maxiter=20*n, callback=cb); out["CG"] = x
    err = {k: (np.linalg.norm(v - xe)/np.linalg.norm(xe) if v is not None else np.nan) for k, v in out.items()}
    res = {k: (np.linalg.norm(H@v - b)/np.linalg.norm(b) if v is not None else np.nan) for k, v in out.items()}
    return np.linalg.cond(H), err, res, it["k"], info
k, err, res, its, info = solve_all(10)
print(f"n=10: kappa={k:.3e}, eps*kappa={np.finfo(float).eps*k:.2e}")
for m in err: print(f"  {m:8s}: rel.err={err[m]:.3e}  rel.residual={res[m]:.3e}")
print(f"  CG iterations={its}, info={info}")
ns = list(range(2, 17)); E = {m: [] for m in ["LU", "Cholesky", "QR", "CG"]}; K = []
for n in ns:
    k, err, res, its, info = solve_all(n); K.append(k)
    for m in E: E[m].append(err[m])
    print(f"n={n:2d} kappa={k:.2e}  " + "  ".join(f"{m}={err[m]:.1e}" for m in E) + f"  (CG it={its})")
plt.semilogy(ns, K, "k--", label=r"$\kappa_2$"); plt.semilogy(ns, np.array(K)*np.finfo(float).eps, "k:", label=r"$\varepsilon_M\kappa$")
for m in E: plt.semilogy(ns, E[m], "o-", label=m)
plt.xlabel("n"); plt.ylabel("relative error"); plt.legend(); plt.savefig("ex_1_4_D.png", dpi=120)
# CG가 '이긴' 이유 점검: 정답 x=1은 H의 큰 고유값 방향에 몰려 있다
n = 10; H = sl.hilbert(n); w, Qv = np.linalg.eigh(H)
c = Qv.T @ np.ones(n)
print("\n|component of x=1| on eigvecs (smallest->largest eig):", np.abs(c).round(6))
print("share of ||1||^2 in top-2 eigvecs:", (c[-2:]**2).sum()/n)
# 무작위 정답 x_rand 로 바꾸면?
rng = np.random.default_rng(0)
for n in [10, 12]:
    H = sl.hilbert(n); xr = rng.normal(size=n); b = H @ xr
    xs = {"LU": sl.lu_solve(sl.lu_factor(H), b), "QR": sl.solve_triangular(*(lambda Q,Rm: (Rm, Q.T@b))(*np.linalg.qr(H))),
          "CG": cg(H, b, rtol=1e-15, atol=0, maxiter=20*n)[0]}
    print(f"random x_true, n={n}: " + "  ".join(f"{m}={np.linalg.norm(v-xr)/np.linalg.norm(xr):.1e}" for m, v in xs.items()))
