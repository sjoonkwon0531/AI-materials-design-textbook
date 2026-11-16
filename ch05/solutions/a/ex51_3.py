"""Ex 5.1.3: memory term -> kappa x(t) as g -> infinity with kappa = b c / g fixed.
a = 1, kappa = -1, b = sqrt(g), c = -sqrt(g), x0 = 1, y0 = 0 (noise-free part).
Also: transient 'noise' impulse b*y0/g -> 0 under this scaling, while an OU-driven y gives
white noise of strength 2 b^2 s^2 / g = 2 |kappa| s^2."""
import numpy as np
from scipy.linalg import expm
a, kap = 1.0, -1.0
tt = np.linspace(0, 6, 601)
print("g      max|x_exact - x_Markov|   (Markov: dx/dt = (-a+kappa) x)")
errs = []
gs = [5, 10, 20, 50, 100, 200, 500]
for g in gs:
    b = np.sqrt(g * abs(kap)); c = -np.sqrt(g * abs(kap))
    A = np.array([[-a, b], [c, -g]])
    xe = np.array([(expm(A * t) @ [1.0, 0.0])[0] for t in tt])
    xm = np.exp((-a + kap) * tt)
    e = np.abs(xe - xm).max(); errs.append(e)
    print(f"{g:5d}  {e:.3e}   impulse std of transient noise b*s/g (s=1) = {b/g:.3f}")
sl = np.polyfit(np.log(gs), np.log(errs), 1)[0]
print(f"slope d ln err / d ln g = {sl:.3f}")
# endpoint weight: int_0^t g e^{-g tau} d tau = 1 - e^{-g t}
for g in (10, 100):
    print(f"g={g}: weight of kernel inside [0,t] at t=0.05: {1-np.exp(-g*0.05):.4f}, at t=0.5: {1-np.exp(-g*0.5):.6f}")
