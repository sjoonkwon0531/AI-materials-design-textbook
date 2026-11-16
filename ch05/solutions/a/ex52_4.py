"""Ex 5.2.4: AR(1) x_{t+1} = phi x_t + sqrt(1-phi^2) xi_t; tau_int and blocking from the lab kernels."""
import sys
import numpy as np
from scipy.signal import lfilter
sys.path.insert(0, "../ch05/labs")
from isingkern import tau_int, blocking

def ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal(n)
    x0 = rng.standard_normal()                           # start in the stationary distribution
    x = lfilter([np.sqrt(1 - phi**2)], [1, -phi], xi, zi=[phi * x0])[0]
    return x

n = 10**6
for phi in (0.95, -0.5):
    x = ar1(phi, n, 0)
    tau_ex = (1 + phi) / (2 * (1 - phi))
    tau, dtau, W, rho = tau_int(x)
    bl = blocking(x)
    print(f"phi={phi}: var={x.var():.4f}, mean={x.mean():+.5f}, rho(1)={rho[1]:.4f} (exact {phi}), rho(5)={rho[5]:+.4f} (exact {phi**5:+.4f})")
    print(f"  tau_int exact = {tau_ex:.4f};  tau_int(Sokal c=6) = {tau:.3f} +/- {dtau:.3f}, window W = {W}")
    print(f"  naive err = {bl[0]:.3e};  sqrt(2 tau_ex / n) = {np.sqrt(2*tau_ex/n):.3e};  sqrt(2 tau_est/n)*sd = {np.sqrt(max(2*tau,0)/n)*x.std():.3e}")
    print("  blocking levels:", " ".join(f"{e:.3e}" for e in bl[:16]))
    nb = [n // 2**k for k in range(len(bl))]
    plat = bl[(np.array(nb) >= 100) & (np.array(nb) <= 2000)]
    print(f"  plateau (levels with 100-2000 blocks): mean = {plat.mean():.3e}; (plateau/naive)^2 = {(plat.mean()/bl[0])**2:.3f} vs 2 tau_ex = {2*tau_ex:.3f}")
# variance of the mean over many independent replicas (direct check), phi = -0.5 and 0.95, n = 10^4
for phi in (-0.5, 0.0, 0.95):
    means = np.array([ar1(phi, 10**4, 100 + s).mean() for s in range(2000)])
    print(f"phi={phi:+.2f}: n*Var(mean) over 2000 replicas (n=1e4) = {1e4*means.var():.3f}; theory 2 tau_int = {(1+phi)/(1-phi):.3f}")
