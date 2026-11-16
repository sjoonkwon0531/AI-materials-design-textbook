"""Ex 9.1.1, 9.1.3: small calculations."""
import numpy as np
from scipy.linalg import expm
print("== 9.1.1 ==")
for d in range(1, 7):
    b = 8 * 100.0**d
    print(f"d={d}: {100**d:.0e} points, {b:.3e} bytes = {b/1e9:.4g} GB = {b/2**30:.4g} GiB; fits in 64 GB: {b <= 64e9}")
for d in range(1, 9):
    print(f"d={d}: grid points x {10**(d/2):.1f} per decade of error (MC: x 100)")
print("== 9.1.3 ==")
g = 0.01; A = np.array([[-g, 1.0], [-1.0, 0.0]])
print("eig", np.linalg.eigvals(A), " -g/2 =", -g/2, " omega =", np.sqrt(1 - g*g/4))
T = 200.0
for x0 in ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0]):
    x = expm(A * T) @ np.array(x0); print(f"x0={x0}: H(T)/H(0) = {(x@x)/(np.dot(x0,x0)):.5f}   exp(-gT) = {np.exp(-g*T):.5f}")
# envelope of H(t)/H0 over one period near T
ts = np.linspace(T - 2*np.pi, T, 400); r = [(lambda x: x@x)(expm(A*t) @ np.array([1.0, 0.0])) / np.exp(-g*t) for t in ts]
print(f"H(t) e^(gt)/H0 over the last period: min {min(r):.5f} max {max(r):.5f}")
print(f"periods in T=200: {T/(2*np.pi):.1f}; lost {100*(1-np.exp(-2)):.1f} %")
# (d) Hamiltonian error dH = g q^2/2
w = np.sqrt(1 + g); print(f"omega = {w:.6f}, phase lag at T=200: {(w-1)*T:.4f} rad = {(w-1)*T/(2*np.pi):.4f} periods; H band [H0, H0(1+g)] -> relative width {g:.3f}")
t = np.linspace(0, 200, 200001); q = np.cos(w*t); p = -w*np.sin(w*t); H = 0.5*(q*q+p*p)
print(f"numerical: H/H0 in [{H.min()/0.5:.5f}, {H.max()/0.5:.5f}], |q - cos t| at T=200: {abs(q[-1]-np.cos(200.0)):.3f}, max over t: {np.abs(q-np.cos(t)).max():.3f}")
