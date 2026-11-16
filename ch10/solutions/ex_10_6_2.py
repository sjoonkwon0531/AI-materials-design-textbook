"""Exercise 10.6.2: gradient descent on E(u) = -alpha u^2/2 + beta u^4/4 (alpha = beta = 1, step eta = 0.1)."""
import numpy as np
al, be, eta = 1.0, 1.0, 0.1
us = np.sqrt(al/be); print(f"u* = {us:.4f}, dE = {-al**2/(4*be):.4f}")
for u0 in (0.0, 1e-3, 1e-6, 1e-12):
    u = u0; n = 0
    while abs(abs(u) - us) > 1e-3 and n < 10000:
        u = u - eta * (-al*u + be*u**3); n += 1
    print(f"u0 = {u0:.0e}: steps {n}, final u {u:.6f}" + ("  (never moves)" if u0 == 0 else
          f"; linear-phase estimate ln(u*/u0)/ln(1+eta*alpha) = {np.log(us/u0)/np.log(1+eta*al):.0f}"))
