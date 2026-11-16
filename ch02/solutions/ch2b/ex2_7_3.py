# Ex 2.7.3: Fourier layer with fixed multiplier R=-k^2 is the spectral Laplacian; a learned R recovers operators
import numpy as np
N = 64; x = 2*np.pi*np.arange(N)/N; k = np.fft.fftfreq(N, d=1.0/N)
u = np.exp(np.sin(x)); upp = (np.cos(x)**2 - np.sin(x))*np.exp(np.sin(x))
fourier_layer = lambda v, R: np.real(np.fft.ifft(R*np.fft.fft(v)))
print(f"R=-k^2 layer vs exact u'': max err = {np.abs(fourier_layer(u, -k**2) - upp).max():.2e}")
# 'learn' R per mode by least squares from input/output pairs of an unknown linear, translation-invariant operator:
# here the exact heat-flow map over time tau: G = exp(tau d^2/dx^2)  ->  true multiplier exp(-k^2 tau)
rng = np.random.default_rng(0); tau = 0.05; M = 20
V = rng.standard_normal((M, N)); W = np.array([fourier_layer(v, np.exp(-k**2*tau)) for v in V])
Vh, Wh = np.fft.fft(V, axis=1), np.fft.fft(W, axis=1)
R_learn = np.sum(np.conj(Vh)*Wh, 0)/np.sum(np.abs(Vh)**2, 0)       # per-mode least squares
print(f"learned R vs exp(-k^2 tau): max |diff| = {np.abs(R_learn - np.exp(-k**2*tau)).max():.2e}")
# with a band limit (FNO keeps only k_max modes), modes above k_max are simply set to zero:
kmax = 12; Rt = np.where(np.abs(k) <= kmax, R_learn, 0)
print(f"truncated to |k|<={kmax}: layer error on a test input = "
      f"{np.abs(fourier_layer(V[0], Rt) - W[0]).max():.2e} (high modes carry exp(-k^2 tau) <= {np.exp(-(kmax+1)**2*tau):.1e})")
