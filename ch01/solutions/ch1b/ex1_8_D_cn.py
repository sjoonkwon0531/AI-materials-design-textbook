"""Ex 1.8.D supplement: Crank-Nicolson reference for u_t = 0.1 u_xx (same problem as the PINN)."""
import numpy as np, time
from scipy.linalg import solve_banded
def cn(Nx, Nt, D=0.1, T=1.0):
    x = np.linspace(0, 1, Nx+1); h = 1/Nx; k = T/Nt; r = D*k/h**2; n = Nx-1
    u = np.sin(np.pi*x[1:-1])
    ab = np.zeros((3, n)); ab[0, 1:] = -r/2; ab[1] = 1 + r; ab[2, :-1] = -r/2
    for _ in range(Nt):
        rhs = (1 - r)*u; rhs[1:] += r/2*u[:-1]; rhs[:-1] += r/2*u[1:]
        u = solve_banded((1, 1), ab, rhs)
    ex = np.exp(-D*np.pi**2*T)*np.sin(np.pi*x[1:-1])
    return np.abs(u-ex).max(), np.linalg.norm(u-ex)/np.linalg.norm(ex)
for N in [20, 50, 100]:
    t = time.perf_counter(); e, rel = cn(N, N); dt = time.perf_counter()-t
    print(f"CN Nx=Nt={N}: max err at t=1 = {e:.2e}, rel L2 = {rel:.2e}, time = {dt*1e3:.1f} ms")
