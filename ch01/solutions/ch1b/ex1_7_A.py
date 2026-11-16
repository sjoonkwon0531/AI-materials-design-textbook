"""Ex 1.7.A: int_0^pi sin = 2 by trapezoid (10 pts), Simpson (11 pts), 2-pt Gauss."""
import numpy as np
from scipy.integrate import trapezoid, simpson
x10 = np.linspace(0, np.pi, 10); T = trapezoid(np.sin(x10), x10)
x11 = np.linspace(0, np.pi, 11); S = simpson(np.sin(x11), x=x11)
h = np.pi/10; S_manual = h/3*(np.sin(x11[0]) + 4*np.sin(x11[1:-1:2]).sum() + 2*np.sin(x11[2:-1:2]).sum() + np.sin(x11[-1]))
xg = np.array([-1, 1])/np.sqrt(3); G = (np.pi/2)*np.sum(np.sin(np.pi/2*(xg+1)))
x11t = x11; T11 = trapezoid(np.sin(x11t), x11t)
for n, v in [("trap 10 pts (9 intervals)", T), ("trap 11 pts", T11), ("Simpson 11 pts", S), ("Simpson manual", S_manual), ("Gauss 2-pt", G)]:
    print(f"{n:28s} {v:.10f}  err={abs(v-2):.3e}")
print("Gauss nodes in [0,pi]:", np.pi/2*(xg+1))
for n in [3, 4, 5]:
    xg, wg = np.polynomial.legendre.leggauss(n); v = (np.pi/2)*np.sum(wg*np.sin(np.pi/2*(xg+1)))
    print(f"Gauss {n}-pt err = {abs(v-2):.3e}")
