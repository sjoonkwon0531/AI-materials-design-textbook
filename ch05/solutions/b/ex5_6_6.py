"""Ex 5.6.6: HMM with a GP-learned constitutive relation D(x) and 3 rounds of active learning."""
import sys, time, numpy as np, warnings
sys.path.insert(0, "../ch05/labs")
import importlib; H = importlib.import_module("56_hmm")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
warnings.filterwarnings("ignore")
t0 = time.time()
M = 32; xf = np.linspace(0, 1, M+1); T, dt = 0.1, 1e-4; snaps = (0.01, 0.1)
fine = H.fv_solve(H.A(np.linspace(0, 1, 2049))*H.a(np.linspace(0, 1, 2049)/H.EPS), 2048, T, dt, snaps=snaps)
coarse = lambda u: u.reshape(M, -1).mean(axis=1)
D_hom = H.A(xf)*H.HARM
def evaluate(D):
    r = H.fv_solve(D, M, T, dt, snaps=snaps)
    e = [np.abs(r[k]-coarse(fine[k])).max() for k in (0.01, 0.1, 'steady')]
    J = D[0]*(1.0-r['steady'][0])/(0.5/M)
    return e, J
WS = 4000*8000                                         # walker-steps per micro run
D33 = np.array([H.micro_D(x, seed=100+k)[0] for k, x in enumerate(xf)])
e, J = evaluate(D33)
print(f"33-point HMM : err t=0.01 {e[0]:.4f}, t=0.1 {e[1]:.4f}, steady {e[2]:.4f}; flux {J:.4f}; micro cost {33*WS:.2e} walker-steps")
e, J = evaluate(D_hom)
print(f"homogenized  : err t=0.01 {e[0]:.4f}, t=0.1 {e[1]:.4f}, steady {e[2]:.4f}; flux {J:.4f}")
idx = [0, 8, 16, 24, 32]                               # x = 0, .25, .5, .75, 1 ; seed rule 100 + k (k = grid index)
Dobs = {k: D33[k] for k in idx}                        # identical to re-running micro_D(x_k, seed=100+k)
def fit(idx):
    X = xf[idx][:, None]; y = np.array([Dobs[k] for k in idx])
    ker = C(1.0, (1e-2, 1e2))*RBF(0.5, (0.05, 10.0)) + WhiteKernel(1e-3, (1e-6, 1e-1))
    gp = GaussianProcessRegressor(ker, normalize_y=True, n_restarts_optimizer=10, random_state=0).fit(X, y)
    mu, sd = gp.predict(xf[:, None], return_std=True)
    return gp, mu, sd
for rnd in range(4):
    gp, mu, sd = fit(idx)
    e, J = evaluate(mu)
    rel = np.abs(mu-D_hom)/D_hom
    print(f"round {rnd}: {len(idx)} micro runs at x={np.round(xf[idx],4).tolist()}")
    print(f"   kernel {gp.kernel_}")
    print(f"   GP mean |rel err| vs homogenized D = {rel.mean():.3f} (max {rel.max():.3f}); max post. std/mean = {(sd/mu).max():.3f}")
    print(f"   macro err t=0.01 {e[0]:.4f}, t=0.1 {e[1]:.4f}, steady {e[2]:.4f}; flux {J:.4f}; micro cost {len(idx)*WS:.2e} walker-steps ({len(idx)/33:.0%} of 33-point HMM)")
    if rnd < 3:
        cand = [k for k in range(M+1) if k not in idx]
        knew = cand[int(np.argmax(sd[cand]))]
        Dobs[knew] = D33[knew]; idx = sorted(idx + [knew])
        print(f"   -> next micro run at x={xf[knew]:.4f} (posterior std {sd[knew]:.4f})")
# variant: noise level fixed from the known micro-estimator scatter (relative std 0.056 from 10 replicas
# at x=0.5 in the Lab), instead of learning it from 5-8 points
print("--- variant: fixed heteroscedastic noise alpha_i = (0.056 y_i)^2, kernel C*RBF, explicit mean removal")
idx = [0, 8, 16, 24, 32]; Dobs = {k: D33[k] for k in idx}
for rnd in range(4):
    X = xf[idx][:, None]; y = np.array([Dobs[k] for k in idx]); ym = y.mean()
    gp = GaussianProcessRegressor(C(0.1, (1e-4, 1e2))*RBF(0.5, (0.05, 10.0)), alpha=(0.056*y)**2,
                                  normalize_y=False, n_restarts_optimizer=10, random_state=0).fit(X, y-ym)
    mu, sd = gp.predict(xf[:, None], return_std=True); mu = mu + ym
    e, J = evaluate(mu); rel = np.abs(mu-D_hom)/D_hom
    print(f"round {rnd}: n={len(idx)}, kernel {gp.kernel_}; |rel err| mean {rel.mean():.3f} max {rel.max():.3f}; "
          f"err {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}; flux {J:.4f}")
    if rnd < 3:
        cand = [k for k in range(M+1) if k not in idx]; knew = cand[int(np.argmax(sd[cand]))]
        Dobs[knew] = D33[knew]; idx = sorted(idx + [knew]); print(f"   -> next x={xf[knew]:.4f} (std {sd[knew]:.4f})")
# HMM's 33 noisy values vs a straight-line fit (what the GP is effectively learning)
p = np.polyfit(xf, D33, 1)
print(f"linear fit to the 33 HMM values: D = {p[0]:.4f} x + {p[1]:.4f}  (exact {H.HARM:.4f} x + {0.5*H.HARM:.4f})")
print(f"wall {time.time()-t0:.0f} s")
