# 연습문제 1.4.E: rotameter 회귀를 5가지 옵티마이저로 (full-batch, 초기값 (10,10))
# 수렴 기준: ||x - x*||_2 < 1e-3 (x* = (1.641, 3.150)), 최대 1e6 step, 학습률은 격자 탐색으로 최적값 사용
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = np.array([10, 30, 50, 70, 90.]); V = np.array([20.0, 52.1, 84.6, 118.3, 151.0]); N = 5
A = np.column_stack([R, np.ones(N)]); xs = np.linalg.lstsq(A, V, rcond=None)[0]
H = 2/N * A.T @ A; lam = np.linalg.eigvalsh(H); L, mu = lam[1], lam[0]
grad = lambda x: 2/N * A.T @ (A @ x - V)
TOL, MAXIT, x0 = 1e-3, 10**6, np.array([10., 10.])
def run(step, maxit=MAXIT, keep=False):
    x = x0.copy(); st = {}; traj = [x.copy()]
    for k in range(1, maxit+1):
        x = step(x, grad(x), st, k)
        if keep and k <= 200000: traj.append(x.copy())
        if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e8: return np.inf, x, traj
        if np.linalg.norm(x - xs) < TOL: return k, x, traj
    return np.nan, x, traj   # nan: 최대 반복 내 미수렴
def gd(lr):
    return lambda x, g, s, k: x - lr*g
def momentum(lr, beta):
    def f(x, g, s, k):
        s["v"] = beta*s.get("v", 0) + g; return x - lr*s["v"]
    return f
def rmsprop(lr, rho=0.9, eps=1e-8):
    def f(x, g, s, k):
        s["E"] = rho*s.get("E", 0) + (1-rho)*g*g; return x - lr*g/(np.sqrt(s["E"])+eps)
    return f
def adam(lr, b1=0.9, b2=0.999, eps=1e-8):
    def f(x, g, s, k):
        s["m"] = b1*s.get("m", 0) + (1-b1)*g; s["v"] = b2*s.get("v", 0) + (1-b2)*g*g
        mh = s["m"]/(1-b1**k); vh = s["v"]/(1-b2**k); return x - lr*mh/(np.sqrt(vh)+eps)
    return f
Hinv = np.linalg.inv(H)
newton = lambda x, g, s, k: x - Hinv @ g
print(f"L={L:.4f}, mu={mu:.6f}, kappa={L/mu:.2f}, x*={xs}")
results = {}
def best(name, makers):
    out = []
    for label, mk in makers:
        k, x, _ = run(mk); out.append((k if np.isfinite(k) else np.inf, label, k))
    out.sort(key=lambda t: t[0])
    print(f"{name:10s}: " + ", ".join(f"[{lab}] -> {('div' if k==np.inf else ('>1e6' if np.isnan(k) else int(k)))}" for _, lab, k in out[:4]))
    results[name] = out[0]
best("GD", [(f"lr={lr:g}", gd(lr)) for lr in [1e-4, 2e-4, 3e-4, 3.02e-4, 2/(L+mu)]])
a_opt = 4/(np.sqrt(L)+np.sqrt(mu))**2; b_opt = ((np.sqrt(L/mu)-1)/(np.sqrt(L/mu)+1))**2
best("Momentum", [(f"lr={a_opt:.3e},beta={b_opt:.4f}(opt)", momentum(a_opt, b_opt))] + [(f"lr={lr:g},beta=0.9", momentum(lr, 0.9)) for lr in [1e-5, 2e-5, 3e-5]] + [(f"lr={lr:g},beta=0.99", momentum(lr, 0.99)) for lr in [1e-6, 2e-6, 3e-6]])
best("RMSprop", [(f"lr={lr:g}", rmsprop(lr)) for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]])
best("Adam", [(f"lr={lr:g}", adam(lr)) for lr in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0]])
for lr in [0.3, 3.0, 30.0]:
    k, x, tr = run(adam(lr), keep=True); tr = np.array(tr)
    print(f"  Adam lr={lr:g}: iters={k}, max excursion |b| along path = {np.abs(tr[:,1]).max():.1f}, |a| = {np.abs(tr[:,0]).max():.1f}")
best("Newton", [("exact Hessian", newton)])
print(f"theory: GD rate (1-2/(kappa+1)) -> iters ~ ln(||e0||/tol)/(2/(kappa+1)) = {np.log(np.linalg.norm(x0-xs)/TOL)*(L/mu+1)/2:.0f}")
print(f"theory: heavy-ball rate (sqrt(k)-1)/(sqrt(k)+1) -> iters ~ {np.log(np.linalg.norm(x0-xs)/TOL)/-np.log((np.sqrt(L/mu)-1)/(np.sqrt(L/mu)+1)):.0f}")
# 궤적 그림 (각 옵티마이저 최선 설정)
fig, ax = plt.subplots(figsize=(6, 5))
aa, bb = np.meshgrid(np.linspace(1.4, 1.9, 300), np.linspace(-2, 11, 300))
JJ = np.mean((aa[..., None]*R + bb[..., None] - V)**2, axis=-1)
ax.contour(aa, bb, np.log10(JJ), 30, colors="0.8", linewidths=.5)
mk = {"GD": gd(3e-4), "Momentum": momentum(a_opt, b_opt), "RMSprop": rmsprop(1e-3), "Adam(lr=0.3)": adam(0.3), "Newton": newton}
for nm, f in mk.items():
    _, _, tr = run(f, keep=True); tr = np.array(tr); ax.plot(tr[:, 0], tr[:, 1], lw=1, label=nm)
ax.set_xlim(1.4, 1.9); ax.set_ylim(-2, 11); ax.plot(*xs, "k*"); ax.set_xlabel("a"); ax.set_ylabel("b"); ax.legend()
fig.savefig("ex_1_4_E.png", dpi=120)
# 왜 적응형(대각 스케일링) 옵티마이저가 잘 듣나: 불량조건이 거의 축 정렬
w, Qh = np.linalg.eigh(H); print("Hessian eigvecs (cols):\n", Qh.round(5))
Dm = np.diag(1/np.sqrt(np.diag(H))); print(f"kappa(D^-1/2 H D^-1/2) (Jacobi-scaled) = {np.linalg.cond(Dm@H@Dm):.3f}")
