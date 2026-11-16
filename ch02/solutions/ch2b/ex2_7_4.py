# Ex 2.7.4: autoregressive instability of a learned linear one-step surrogate for 1D Cahn-Hilliard data
import numpy as np
N, L, eps = 128, 4*np.pi, 0.1
k = np.fft.fftfreq(N, d=L/N)*2*np.pi; K2 = k**2; K4 = K2**2

def ch1d(c, T, dt=0.005, S=2.0, every=None):
    den = 1 + dt*S*K2 + dt*eps**2*K4; gain = 1 + dt*S*K2; ch = np.fft.fft(c); out = [c.copy()]
    n_every = int(round(every/dt))
    for n in range(1, int(round(T/dt))+1):
        ch = (gain*ch - dt*K2*np.fft.fft(c**3 - c))/den; c = np.real(np.fft.ifft(ch))
        if n % n_every == 0: out.append(c.copy())
    return np.array(out)

Dt = 0.1                                    # surrogate time step
rng = np.random.default_rng(0)
X, Y = [], []
for traj in range(40):                      # training data: 40 quenches, t in [0, 20]
    c0 = 0.1*(2*rng.random(N) - 1)
    snaps = ch1d(c0, 20.0, every=Dt)
    X.append(snaps[:-1]); Y.append(snaps[1:])
X = np.vstack(X); Y = np.vstack(Y)
A = np.linalg.lstsq(X, Y, rcond=None)[0].T            # c_{n+1} ~ A c_n
ev = np.linalg.eigvals(A)
print(f"training pairs: {len(X)};  one-step fit residual (rel) = {np.linalg.norm(X@A.T - Y)/np.linalg.norm(Y):.2e}")
print(f"rho(A) = {np.abs(ev).max():.4f};  #eigenvalues with |mu|>1: {(np.abs(ev) > 1+1e-9).sum()}")

c0 = 0.1*(2*np.random.default_rng(123).random(N) - 1)  # unseen test quench
true = ch1d(c0, 105.0, every=Dt)[50:]          # start the rollout from the t=5 state (patterned)
c0 = true[0]
def rollout(A, c, n):
    out = [c]
    for _ in range(n): out.append(A@out[-1])
    return np.array(out)
pred = rollout(A, c0, len(true)-1)
for n in [1, 10, 50, 100, 200, 500, 1000]:
    e = np.abs(pred[n] - true[n]).max()
    print(f"rollout step {n:5d} (t=5+{n*Dt:6.1f}): max|c_pred| = {np.abs(pred[n]).max():.3e}  max err = {e:.3e}")
# growth rate of the error vs rho(A)
n1, n2 = 500, 1000; rate = (np.abs(pred[n2]).max()/np.abs(pred[n1]).max())**(1/(n2-n1))
print(f"late growth factor per step = {rate:.4f}  vs rho(A) = {np.abs(ev).max():.4f}")
# remedy: clip the spectrum to the unit disk (spectral normalisation analogue)
w, V = np.linalg.eig(A); w_c = np.where(np.abs(w) > 1, w/np.abs(w), w)
Ac = np.real(V@np.diag(w_c)@np.linalg.inv(V))
pc = rollout(Ac, c0, len(true)-1)
print(f"clipped rho = {np.abs(np.linalg.eigvals(Ac)).max():.4f}:  max|c_pred| at step 1000 = {np.abs(pc[1000]).max():.3f}, "
      f"max err = {np.abs(pc[1000]-true[1000]).max():.3f}  (true max|c| = {np.abs(true[1000]).max():.3f})")
print("  ||Ac^n||_2 for n=10,100,1000 (non-normal transient):", [round(np.linalg.norm(np.linalg.matrix_power(Ac, n), 2), 2) for n in (10, 100, 1000)])
print("  max|c_pred| (clipped) at steps 100, 2000:", round(np.abs(pc[100]).max(), 3), round(np.abs(rollout(Ac, c0, 2000)[-1]).max(), 3))
# a purely linear map cannot saturate: what the eigenvalues >1 represent
print("modes with |mu|>1 (|mu|, dominant |k| of eigenvector):",
      [(round(abs(w[i]), 4), round(float(np.abs(k)[np.abs(np.fft.fft(np.real(V[:, i]))).argmax()]), 2)) for i in np.argsort(-np.abs(w))[:4]])
print(f"continuous linear theory: sigma(k*) = 1/(4 eps^2) = {1/(4*eps**2):.1f} -> exp(sigma Dt) = {np.exp(0.25/eps**2*Dt):.2f}")
# stronger remedy: singular-value clipping (true spectral normalisation): ||A_s||_2 <= 1  =>  ||A_s^n|| <= 1
U_, s_, Vt_ = np.linalg.svd(A); As = U_@np.diag(np.minimum(s_, 1.0))@Vt_
ps = rollout(As, c0, 2000)
print(f"sigma_max(A) = {s_.max():.3f} (non-normal: >> rho(A));  SV-clipped: rho = {np.abs(np.linalg.eigvals(As)).max():.4f}, "
      f"max|c_pred| at steps 100/1000/2000 = {np.abs(ps[100]).max():.3f}/{np.abs(ps[1000]).max():.3f}/{np.abs(ps[2000]).max():.3f}, "
      f"err at 1000 = {np.abs(ps[1000]-true[1000]).max():.3f}")
