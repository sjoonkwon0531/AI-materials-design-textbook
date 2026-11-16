# Ex 2.4.7: spectral 2nd and 4th derivatives of u=exp(sin x) on [0,2pi)
import numpy as np
import sympy as sp
X = sp.Symbol('x'); U = sp.exp(sp.sin(X))
d2 = sp.lambdify(X, sp.diff(U, X, 2), 'numpy'); d4 = sp.lambdify(X, sp.diff(U, X, 4), 'numpy')
u = lambda x: np.exp(np.sin(x))

def spec(N, p):
    x = 2*np.pi*np.arange(N)/N
    k = np.fft.fftfreq(N, d=1.0/N)
    if p % 2 == 1: k[N//2] = 0          # odd derivatives: zero the Nyquist mode (N even)
    return x, np.real(np.fft.ifft((1j*k)**p*np.fft.fft(u(x))))   # one line: (ik)^p * u_hat

def fdm2(N):
    x = 2*np.pi*np.arange(N)/N; h = 2*np.pi/N; uh = u(x)
    return x, (np.roll(uh, -1) - 2*uh + np.roll(uh, 1))/h**2

print("   N   spec u'' err   spec u'''' err   FDM2 u'' err")
for N in [8, 12, 16, 24, 32, 48, 64]:
    x, s2 = spec(N, 2); _, s4 = spec(N, 4); _, f2 = fdm2(N)
    print(f"{N:4d}   {np.abs(s2-d2(x)).max():.3e}      {np.abs(s4-d4(x)).max():.3e}      {np.abs(f2-d2(x)).max():.3e}")
print("max |u''''| on grid =", f"{np.abs(d4(2*np.pi*np.arange(64)/64)).max():.3f}")
# round-off floor estimate: eps_mach * (N/2)^p * max|u|
for N in [32, 64]:
    print(f"N={N}: round-off floor estimate p=2: {np.finfo(float).eps*(N/2)**2*np.e:.1e},  p=4: {np.finfo(float).eps*(N/2)**4*np.e:.1e}")
