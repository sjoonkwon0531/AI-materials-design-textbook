# Shared helpers: Lab 2.14 Cahn-Hilliard (stabilized semi-implicit spectral) + diagnostics
import numpy as np

def grid(N=256, L=4*np.pi):
    k1 = np.fft.fftfreq(N, d=L/N)*2*np.pi
    KX, KY = np.meshgrid(k1, k1, indexing='ij'); K2 = KX**2 + KY**2
    return k1, KX, KY, K2

def ch_run(N=256, L=4*np.pi, eps=0.10, dt=0.05, S=2.0, T=400.0, c0=None, seed=1, mean=0.0,
           callback=None, every=20, explicit=False):
    """Lab 2.14 scheme (Box 2.22). explicit=True -> forward Euler with S=0 (biharmonic explicit too)."""
    k1, KX, KY, K2 = grid(N, L); K4 = K2**2
    if c0 is None:
        c = mean + 0.1*(2*np.random.default_rng(seed).random((N, N)) - 1)
    else:
        c = c0.copy()
    chat = np.fft.fft2(c)
    if explicit:
        lin = 1 - dt*eps**2*K4
    else:
        denom = 1 + dt*S*K2 + dt*eps**2*K4; gain = 1 + dt*S*K2
    nsteps = int(round(T/dt))
    for n in range(1, nsteps+1):
        nl = np.fft.fft2(c**3 - c)
        if explicit:
            chat = lin*chat - dt*K2*nl
        else:
            chat = (gain*chat - dt*K2*nl)/denom
        c = np.real(np.fft.ifft2(chat))
        if callback is not None and n % every == 0:
            if callback(n*dt, c, chat) is False: break
    return c

def free_energy(c, L=4*np.pi, eps=0.10):
    N = c.shape[0]; k1, KX, KY, K2 = grid(N, L); ch = np.fft.fft2(c); dA = (L/N)**2
    cx = np.real(np.fft.ifft2(1j*KX*ch)); cy = np.real(np.fft.ifft2(1j*KY*ch))
    return np.sum(0.25*(c**2-1)**2 + 0.5*eps**2*(cx**2+cy**2))*dA

def grad_mu_sq(c, L=4*np.pi, eps=0.10):
    N = c.shape[0]; k1, KX, KY, K2 = grid(N, L); dA = (L/N)**2
    muh = np.fft.fft2(c**3 - c) + eps**2*K2*np.fft.fft2(c)
    mx = np.real(np.fft.ifft2(1j*KX*muh)); my = np.real(np.fft.ifft2(1j*KY*muh))
    return np.sum(mx**2 + my**2)*dA

def length_scale(c, L=4*np.pi):
    """L(t) = 2 pi / <k>_S(k), S(k)=|c_hat|^2 (k=0 excluded), as in Lab 2.14."""
    N = c.shape[0]; k1, KX, KY, K2 = grid(N, L); Sk = np.abs(np.fft.fft2(c - c.mean()))**2
    k = np.sqrt(K2); m = k > 0
    return 2*np.pi/(np.sum(k[m]*Sk[m])/np.sum(Sk[m]))

def peak_k(c, L=4*np.pi):
    """radially averaged structure-factor peak (bin width 2pi/L)."""
    N = c.shape[0]; k1, KX, KY, K2 = grid(N, L); Sk = np.abs(np.fft.fft2(c - c.mean()))**2
    dk = 2*np.pi/L; kb = np.rint(np.sqrt(K2)/dk).astype(int)
    prof = np.bincount(kb.ravel(), Sk.ravel())/np.maximum(np.bincount(kb.ravel()), 1)
    prof[0] = 0; return prof.argmax()*dk
