"""Ex 1.7.D: DOS of 1D tight binding eps=-2t cos k (t=a=1), per site per spin, normalized to 1.
(a) Monkhorst-Pack mesh Nk=100 + Gaussian smearing; (b) 1D linear 'tetrahedron' (segment) method."""
import numpy as np
from scipy.special import erf
Nk = 100
g_exact = lambda E: np.where(np.abs(E) < 2, 1/(np.pi*np.sqrt(np.clip(4-E**2, 1e-300, None))), 0.0)
N_exact = lambda E: np.where(E <= -2, 0.0, np.where(E >= 2, 1.0, np.arccos(np.clip(-E/2, -1, 1))/np.pi))
j = np.arange(1, Nk+1); k_mp = np.pi*(2*j - Nk - 1)/Nk          # MP mesh in [-pi, pi)
e_mp = -2*np.cos(k_mp)
def dos_mp(E, sigma):
    return np.mean(np.exp(-(E[:, None]-e_mp[None, :])**2/(2*sigma**2)), axis=1)/(np.sqrt(2*np.pi)*sigma)
def N_mp(E, sigma=None):
    if sigma is None: return np.mean(e_mp[None, :] <= E[:, None], axis=1)
    return np.mean(0.5*(1+erf((E[:, None]-e_mp[None, :])/(np.sqrt(2)*sigma))), axis=1)
# segment ("tetrahedron") method on uniform k grid with Nk segments over the BZ
kk = np.linspace(-np.pi, np.pi, Nk+1); ee = -2*np.cos(kk)
lo, hi = np.minimum(ee[:-1], ee[1:]), np.maximum(ee[:-1], ee[1:])
def dos_tet(E):
    inside = (E[:, None] > lo) & (E[:, None] < hi)
    return np.sum(inside/(hi-lo), axis=1)/Nk
def N_tet(E):
    frac = np.clip((E[:, None]-lo)/(hi-lo), 0, 1)
    return np.sum(frac, axis=1)/Nk
for EF in [0.0, 1.0, -1.9]:
    E = np.array([EF + 1e-9])   # avoid landing exactly on a node energy
    print(f"E_F={EF:+.2f}: exact g={g_exact(E)[0]:.6f}, N={N_exact(E)[0]:.6f}")
    for s in [0.02, 0.05, 0.1]:
        print(f"   MP sigma={s}: g={dos_mp(E,s)[0]:.6f} (rel err {abs(dos_mp(E,s)[0]/g_exact(E)[0]-1):.1e}), N={N_mp(E,s)[0]:.6f}")
    print(f"   MP step count: N={N_mp(E)[0]:.6f}")
    print(f"   tetra: g={dos_tet(E)[0]:.6f} (rel err {abs(dos_tet(E)[0]/g_exact(E)[0]-1):.1e}), N={N_tet(E)[0]:.6f} (err {abs(N_tet(E)[0]-N_exact(E)[0]):.1e})")
# fine scan of N(E) around E_F=0: step artifacts
E = np.linspace(-0.2, 0.2, 4001)
print(f"max |N_MP_step - N_exact| on [-0.2,0.2] = {np.abs(N_mp(E)-N_exact(E)).max():.2e}")
print(f"max |N_tet - N_exact|      on [-0.2,0.2] = {np.abs(N_tet(E)-N_exact(E)).max():.2e}")
print(f"max |g_tet/g_exact - 1|    on [-0.2,0.2] = {np.abs(dos_tet(E)/g_exact(E)-1).max():.2e}")
print(f"max |g_MP(0.05)/g_ex - 1|  on [-0.2,0.2] = {np.abs(dos_mp(E,0.05)/g_exact(E)-1).max():.2e}")
print(f"max |g_MP(0.02)/g_ex - 1|  on [-0.2,0.2] = {np.abs(dos_mp(E,0.02)/g_exact(E)-1).max():.2e}")
print(f"MP level spacing near E=0: {2*np.pi/Nk*2:.4f}  (dE = |de/dk| dk = 2*(2pi/Nk))")
