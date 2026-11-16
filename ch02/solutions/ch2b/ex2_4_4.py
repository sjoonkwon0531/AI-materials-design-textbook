# Ex 2.4.4: Neumann BC enters the weak form as g*v(1) (natural BC) -- P1 FEM check
import numpy as np
g = 0.5
exact = lambda x: -x**2/2 + (1+g)*x          # -u''=1, u(0)=0, u'(1)=g

def fem(N, use_g=True):
    h = 1.0/N; x = np.linspace(0, 1, N+1)
    K = np.zeros((N+1, N+1)); b = np.zeros(N+1)
    for e in range(N):
        K[e:e+2, e:e+2] += np.array([[1, -1], [-1, 1]])/h
        b[e:e+2] += 0.5*h*1.0                 # f=1: int f phi_i over element = h/2
    if use_g: b[N] += g                       # boundary term [u' v]_0^1 -> g v(1)
    # essential BC u(0)=0: remove row/col 0 (v(0)=0 in test space)
    u = np.zeros(N+1); u[1:] = np.linalg.solve(K[1:, 1:], b[1:])
    return x, u

for N in [4, 16, 64]:
    x, u = fem(N); err = np.abs(u-exact(x)).max()
    slope = (u[-1]-u[-2])/(x[-1]-x[-2])
    print(f"N={N:3d}: nodal max err={err:.2e}  u_h(1)={u[-1]:.6f} (exact {exact(1.0):.6f})  last-element slope={slope:.4f} (g={g})")
x, u = fem(64, use_g=False)
print(f"without g*v(1) term: u_h(1)={u[-1]:.6f}  -> solves u'(1)=0 problem (exact {0.5:.6f})")
