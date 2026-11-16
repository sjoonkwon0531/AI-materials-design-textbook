# Ex 2.4.6: FVM with central flux F_{i+1/2}=a(u_i+u_{i+1})/2 + forward Euler (= FTCS for advection)
import numpy as np

def run(N=200, a=1.0, CFL=0.8, tfinal=1.0, flux='central'):
    dx = 1.0/N; xc = (np.arange(N)+0.5)*dx
    u = np.where((xc > 0.3) & (xc < 0.6), 1.0, 0.0)
    dt = CFL*dx/a; nu = a*dt/dx; nsteps = int(round(tfinal/dt))
    m0 = u.sum()*dx; drift = 0.0; rdrift = 0.0; hist = []
    for n in range(nsteps):
        F = a*0.5*(u + np.roll(u, -1)) if flux == 'central' else a*u   # F at face i+1/2
        u = u - dt/dx*(F - np.roll(F, 1))                               # telescoping form
        drift = max(drift, abs(u.sum()*dx - m0)/m0); rdrift = max(rdrift, abs(u.sum()*dx - m0)/(np.abs(u).sum()*dx))
        hist.append(np.abs(u).max())
    return u, (drift, rdrift), np.array(hist), nsteps

for CFL in [0.8, 0.1]:
    for fl in ['upwind', 'central']:
        u, d, h, n = run(CFL=CFL, flux=fl)
        print(f"CFL={CFL:3.1f} {fl:8s} steps={n:5d}  max|u|(t=1)={h[-1]:.3e}  mass drift/m0={d[0]:.1e}  drift/sum|u|={d[1]:.1e}")

# von Neumann: central (FTCS) g = 1 - i nu sin(theta) -> |g|^2 = 1 + nu^2 sin^2(theta) > 1 always
nu = 0.8; th = np.linspace(0, np.pi, 1001); gmax = np.sqrt(1+nu**2*np.sin(th)**2).max()
print(f"\nvon Neumann central/FTCS, nu=0.8: max|g| = {gmax:.4f} (theta=pi/2);  most amplified mode = 4-cell wavelength")
for N in [200]:
    u, d, h, n = run(N=N, CFL=0.8, flux='central')
    # measured growth rate in the late phase
    rate = (h[-1]/h[n//2])**(1/(n-n//2))
    print(f"measured late-time growth factor of max|u| per step = {rate:.4f}  (bound {gmax:.4f})")
