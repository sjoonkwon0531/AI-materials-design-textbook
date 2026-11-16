# Ex 2.6.5: Allen-Cahn (non-conserved) with the same stabilized semi-implicit spectral frame
#   c^{n+1}_hat = ((1+dt S) c^n_hat - dt FFT(c^3-c)) / (1 + dt S + dt eps^2 k^2)
import numpy as np
from ch_common import grid, length_scale
from ex2_6_5_tools import interface_length_scale

def ac_run(N, Lbox, eps, dt, T, S=2.0, seed=1, times=()):
    k1, KX, KY, K2 = grid(N, Lbox)
    c = 0.1*(2*np.random.default_rng(seed).random((N, N)) - 1); ch = np.fft.fft2(c)
    denom = 1 + dt*S + dt*eps**2*K2
    m0 = c.mean(); out = {}; tset = {int(round(t/dt)): t for t in times}
    for n in range(1, int(round(T/dt))+1):
        ch = ((1 + dt*S)*ch - dt*np.fft.fft2(c**3 - c))/denom
        c = np.real(np.fft.ifft2(ch))
        if n in tset: out[tset[n]] = (interface_length_scale(c, Lbox), length_scale(c, Lbox), c.mean())
    return m0, out

N, Lbox, eps, dt = 512, 64.0, 0.25, 0.1           # dx = 0.125 = eps/2, box = 256 eps
times = [5, 10, 20, 40, 80, 160, 320, 640]
m0, out = ac_run(N, Lbox, eps, dt, 640.0, times=times)
print(f"Allen-Cahn: N={N}, box={Lbox}, eps={eps}, dt={dt}, S=2;  initial mean = {m0:+.2e}")
print("     t    L_int=A/P   L_int^2/t   L_int^3/t    2pi/<k>     mean c   |mean-mean0|")
for t in times:
    Li, Lk, m = out[t]
    print(f"{t:6g}    {Li:7.3f}     {Li**2/t:.4f}      {Li**3/t:.4f}    {Lk:7.3f}   {m:+.4f}   {abs(m-m0):.2e}")
for lo, hi in [(20, 640), (40, 640)]:
    tt = np.array([t for t in times if lo <= t <= hi])
    for j, lab in [(0, 'L_int'), (1, '2pi/<k>')]:
        LL = np.array([out[t][j] for t in tt])
        print(f"log-log slope {lab:8s} over {lo}<=t<={hi}: n = {np.polyfit(np.log(tt), np.log(LL), 1)[0]:.3f}")
print("theory: Allen-Cahn n = 1/2 ; Cahn-Hilliard (Lab 2.14) n = 1/3")
