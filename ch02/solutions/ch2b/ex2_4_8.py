# Ex 2.4.8: Gibbs phenomenon -- spectral (Fourier) approximation of smooth / kink / jump functions
#   P_N : truncated Fourier series (projection, |k| < N/2)        -> classical Gibbs overshoot 8.95%
#   I_N : trigonometric interpolant on N equispaced nodes (FFT)    -> overshoot depends on node placement
import numpy as np
from scipy.special import sici
M = 2**16
xf = 2*np.pi*np.arange(M)/M
def square(x, shift=0.0):           # jump of size 1 at x=shift and x=pi+shift, value 1/2 exactly at jumps
    y = (x - shift) % (2*np.pi)
    return np.where(np.isclose(y, 0) | np.isclose(y, np.pi), 0.5, (y < np.pi).astype(float))
funcs = {'smooth exp(sin x)': lambda x: np.exp(np.sin(x)),
         'kink |sin x|': lambda x: np.abs(np.sin(x)),
         'jump square': square}

def to_fine(uh, N):                 # place N-mode spectrum (normalized) on the fine grid
    big = np.zeros(M, complex)
    big[:N//2] = uh[:N//2]; big[-N//2+1:] = uh[-N//2+1:]
    big[N//2] = 0.5*uh[N//2]; big[-N//2] = 0.5*uh[N//2]
    return np.real(np.fft.ifft(big))*M
def interp(fn, N):
    x = 2*np.pi*np.arange(N)/N; return to_fine(np.fft.fft(fn(x))/N, N)
def project(fn, N):
    uh = np.fft.fft(fn(xf))/M; t = np.zeros(N, complex)
    t[:N//2] = uh[:N//2]; t[-N//2+1:] = uh[-N//2+1:]; t[N//2] = 0
    return to_fine(t, N)
fit = lambda N, e: np.polyfit(np.log(N), np.log(e), 1)[0]
far = (xf > np.pi/4) & (xf < 3*np.pi/4)          # region at distance >= pi/4 from both jumps

Ns = np.array([16, 32, 64, 128, 256, 512, 1024])
for name, fn in funcs.items():
    uf = fn(xf)
    for meth, A in [('projection P_N', project), ('interpolant I_N', interp)]:
        mx, l2, ff, ov = [], [], [], []
        for N in Ns:
            e = A(fn, N) - uf
            mx.append(np.abs(e).max()); l2.append(np.sqrt(np.mean(e**2)*2*np.pi))
            ff.append(np.abs(e[far]).max()); ov.append((e+uf).max() - uf.max())
        print(f"\n[{name} | {meth}]\n    N    max err    L2 err     far-field err  overshoot")
        for i, N in enumerate(Ns):
            print(f"{N:5d}  {mx[i]:.3e}  {l2[i]:.3e}  {ff[i]:.3e}      {ov[i]:+.4f}")
        if name != 'smooth exp(sin x)':
            print(f"  log-log slopes N=64..1024: max {fit(Ns[2:], mx[2:]):+.2f}, L2 {fit(Ns[2:], l2[2:]):+.2f}, far-field {fit(Ns[2:], ff[2:]):+.2f}")

print("\njump between nodes (shift = half grid spacing), interpolant overshoot:")
for N in [64, 256, 1024]:
    fn = lambda x, N=N: square(x, np.pi/N)
    print(f"  N={N:5d}: overshoot = {interp(fn, N).max()-1:+.4f}")

print("\n|u_hat_k| (Box 2.15 decay):")
N = 4096; x = 2*np.pi*np.arange(N)/N
for name, ks in [('smooth exp(sin x)', [4, 8, 16, 24]), ('kink |sin x|', [10, 100, 1000]), ('jump square', [9, 99, 999])]:
    uh = np.abs(np.fft.fft(funcs[name](x))/N)
    print(f"  {name:18s}: " + "  ".join(f"k={k}: {uh[k]:.2e}" for k in ks))
print(f"\nGibbs constant Si(pi)/pi - 1/2 = {sici(np.pi)[0]/np.pi-0.5:.4f}")

try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3)); sel = (xf > np.pi-0.6) & (xf < np.pi+0.6)
    ax.plot(xf[sel], square(xf)[sel], 'k-', lw=1, label='exact')
    for N in [32, 128]: ax.plot(xf[sel], project(square, N)[sel], label=f'P_N, N={N}')
    ax.legend(); fig.tight_layout(); fig.savefig('ex2_4_8_gibbs.png', dpi=120); print("saved ex2_4_8_gibbs.png")
except Exception as ex:
    print("figure skipped:", ex)
