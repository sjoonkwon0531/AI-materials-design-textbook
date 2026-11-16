# Ex 4.5.3: learning curve (Lab 4.5 KRR) + energy-only vs energy+force matching (differentiable RBF-basis model)
import numpy as np
def U_true(r): return 4.0*(r**-12 - r**-6)
def F_true(r): return 48.0*r**-13 - 24.0*r**-7
rg = np.linspace(0.95, 2.5, 400)
rmse = lambda a,b: np.sqrt(np.mean((a-b)**2))
# --- (1) reproduce Lab 4.5 learning curve (same seed / sampler / KRR) ---
class KRR_U:
    def __init__(s, gamma=2.0, lam=1e-8): s.gamma, s.lam = gamma, lam
    def fit(s, r, y):
        s.r = r.copy(); K = np.exp(-s.gamma*(r[:,None]-r[None,:])**2)
        s.alpha = np.linalg.solve(K + s.lam*np.eye(len(r)), y); return s
    def U(s, q): return np.exp(-s.gamma*(q[:,None]-s.r[None,:])**2) @ s.alpha
    def F(s, q):
        d = q[:,None]-s.r[None,:]; return -((np.exp(-s.gamma*d**2)*(-2*s.gamma*d)) @ s.alpha)
np.random.seed(2)
print("(1) Lab 4.5 learning curve (KRR, energy only)\n  N    U-RMSE     F-RMSE")
for n in [10,20,40,80,160,320]:
    r = np.random.uniform(0.92,2.5,n); m = KRR_U().fit(r,U_true(r))
    print(f"  {n:3d}  {rmse(m.U(rg),U_true(rg)):.3e}  {rmse(m.F(rg),F_true(rg)):.3e}")
# --- (2) differentiable linear model U(r)=sum_k w_k phi_k(r), phi_k Gaussian RBF ---
C = np.linspace(0.85, 2.6, 60); g = 300.0          # 60 centres, width ~0.04
def Phi(r):  return np.exp(-g*(r[:,None]-C[None,:])**2)
def dPhi(r): d=r[:,None]-C[None,:]; return -2*g*d*np.exp(-g*d**2)
def fit(r, lamF, lam=1e-8):
    A = [Phi(r)]; b = [U_true(r)]
    if lamF>0: A.append(np.sqrt(lamF)*(-dPhi(r))); b.append(np.sqrt(lamF)*F_true(r))
    A = np.vstack(A); b = np.concatenate(b)
    return np.linalg.solve(A.T@A + lam*np.eye(len(C)), A.T@b)
print("\n(2) RBF-basis model, median over 20 random training sets, test on [0.95,2.5]")
print("  N   |  F-RMSE (E only)  F-RMSE (E+F, lamF=1)  |  U-RMSE (E only)  U-RMSE (E+F)  | F-RMSE (E only, 2N)")
rng = np.random.default_rng(0)
for n in [5,10,20,40,80]:
    res = []
    for t in range(20):
        r = rng.uniform(0.92,2.5,n); out=[]
        for lamF in [0.0, 1.0]:
            w = fit(r, lamF); out += [rmse(-dPhi(rg)@w, F_true(rg)), rmse(Phi(rg)@w, U_true(rg))]
        r2 = rng.uniform(0.92,2.5,2*n); w = fit(r2, 0.0)       # energy-only with 2N labels (same #equations)
        out += [rmse(-dPhi(rg)@w, F_true(rg))]
        res.append(out)
    m = np.median(res, axis=0)
    print(f"  {n:3d} |  {m[0]:10.3e}       {m[2]:10.3e}          |  {m[1]:10.3e}     {m[3]:10.3e}   |  {m[4]:10.3e}")
print("  (values are medians over the 20 training sets)")
