"""Ex 5.3.4: dimerisation 2A -> B, propensity c A(A-1)/2; SSA (direct method) vs mean-field ODE A0/(1+c A0 t)."""
import numpy as np
from numba import njit

@njit(cache=True)
def ssa_dimer(A0, c, tobs, seed):
    np.random.seed(seed)
    A = A0; t = 0.0
    while True:
        a = c * A * (A - 1) / 2.0
        if a <= 0.0:
            return A
        tau = -np.log(np.random.random()) / a
        if t + tau > tobs:
            return A
        t += tau; A -= 2

for A0, c in [(10, 0.1), (1000, 1e-3)]:
    ntraj = 10000
    vals = np.array([ssa_dimer(A0, c, 1.0, 1000 + s) for s in range(ntraj)], float)
    ode = A0 / (1 + c * A0 * 1.0)
    m = vals.mean(); se = vals.std(ddof=1) / np.sqrt(ntraj)
    print(f"A0={A0}, c={c}: <A(1)>_SSA = {m:.4f} +/- {se:.4f}, ODE = {ode:.4f}, rel. diff = {(m-ode)/ode*100:+.2f} %,"
          f" Var A(1) = {vals.var(ddof=1):.3f}, <A^2>-<A>^2 / <A> = {vals.var(ddof=1)/m:.3f}")
    # moment check: d<A>/dt = -c <A(A-1)>; compare <A(A-1)> with <A>^2 at t = 1
    print(f"   at t=1: <A(A-1)> = {np.mean(vals*(vals-1)):.3f}, <A>^2 = {m**2:.3f}, <A^2> = {np.mean(vals**2):.3f}")
# exact master-equation mean for A0=10 via matrix exponential, as a check of the SSA
from scipy.linalg import expm
A0, c = 10, 0.1
states = np.arange(A0, -1, -2)                          # 10, 8, ..., 0
K = np.zeros((len(states), len(states)))
for i, A in enumerate(states):
    r = c * A * (A - 1) / 2
    K[i, i] -= r
    if i + 1 < len(states):
        K[i + 1, i] += r
P = expm(K * 1.0) @ np.eye(len(states))[:, 0]
print(f"master equation (exact) <A(1)> for A0=10: {P @ states:.4f}")
# exact master equation for A0=1000 (states 1000..0 step 2) as well
A0, c = 1000, 1e-3
states = np.arange(A0, -1, -2)
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
r = c * states * (states - 1) / 2
K = diags([-r, r[:-1]], [0, -1], format="csc")
p0 = np.zeros(len(states)); p0[0] = 1
P = expm_multiply(K, p0, start=0, stop=1.0, num=2, endpoint=True)[-1]
print(f"master equation (exact) <A(1)> for A0=1000: {P @ states:.4f}")
