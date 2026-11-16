# Ex 4.6.5: toy demonstrations of ENCUT (plane-wave cutoff) and k-point/smearing convergence
import numpy as np
from scipy.special import erfc
# (a) 1D plane-wave basis, periodic cell L, Gaussian well V(x) = -V0 exp(-x^2/(2 s^2)); hbar=m=1
L, V0, s = 10.0, 10.0, 0.3
xg = np.linspace(-L/2, L/2, 4096, endpoint=False); Vx = -V0*np.exp(-xg**2/(2*s**2))
Vq = np.fft.fft(Vx)/len(xg)                                   # Fourier coefficients V_G
def e0(Ecut):
    nmax = int(np.floor(np.sqrt(2*Ecut)*L/(2*np.pi))); n = np.arange(-nmax, nmax+1); G = 2*np.pi*n/L
    H = np.diag(0.5*G**2) + Vq[(n[:,None]-n[None,:]) % len(xg)].real
    return len(n), np.linalg.eigvalsh(H)[0]
print("(a) plane-wave cutoff: ground-state energy (variational, decreases monotonically)")
ref = e0(4000)[1]
for Ec in [5, 10, 20, 40, 80, 160, 320]:
    nb, e = e0(Ec); print(f"   Ecut={Ec:4d}  #PW={nb:4d}  E0={e:+.6f}  E0-E0(Ecut=4000)={e-ref:.2e}")
# (b) 1D tight-binding metal (eps_k = -2t cos k, t=1), half filling: band energy per site vs k-mesh
def band_energy(Nk, sigma):
    k = 2*np.pi*(np.arange(Nk)+0.37)/Nk; ek = -2*np.cos(k)         # shifted mesh: Fermi points generically off-grid (as in 2D/3D)
    def nel(mu): return np.mean(0.5*erfc((ek-mu)/sigma)) if sigma>0 else np.mean(np.where(ek<mu,1.0,np.where(ek==mu,0.5,0.0)))
    lo, hi = -3, 3
    for _ in range(200):
        mu = 0.5*(lo+hi); lo, hi = (mu, hi) if nel(mu) < 0.5 else (lo, mu)
    occ = 0.5*erfc((ek-mu)/sigma) if sigma>0 else (ek<mu)+0.5*(ek==mu)
    if sigma == 0:                                                     # fix electron count exactly at degeneracy
        occ = occ.astype(float); deg = np.isclose(ek, mu)
        if deg.any(): occ[deg] = (0.5*Nk - occ[~deg].sum())/deg.sum()
    return np.mean(occ*ek)
exact = -2/np.pi
lim = {sg: (exact if sg==0 else band_energy(20000, sg)) for sg in [0.0,0.1,0.3]}
print("(b) k-mesh convergence: |E(Nk,sigma) - E(Nk->inf,sigma)| per site (t=1; sigma=0 limit = -2/pi)")
print("   Nk     sigma=0       sigma=0.1     sigma=0.3")
for Nk in [7, 12, 18, 25, 41, 64, 101, 160, 250, 400]:
    print(f"  {Nk:4d}  " + "  ".join(f"{abs(band_energy(Nk,sg)-lim[sg]):.2e}" for sg in [0.0,0.1,0.3]))
for sg in [0.1,0.3]:
    print(f"   sigma={sg}: converged (Nk=20000) value differs from exact by {abs(band_energy(20000,sg)-exact):.2e}  (smearing bias)")
