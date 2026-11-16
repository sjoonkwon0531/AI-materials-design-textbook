# Ex 2.3.B (Baumgarte alpha/beta sweep) and 2.3.C (projection vs Baumgarte)
import numpy as np
from scipy.integrate import solve_ivp
g = 9.81

def rhs(t, s, a=None, b=None):
    x, y, vx, vy = s
    r2 = x*x + y*y; v2 = vx*vx + vy*vy
    phi = r2 - 1.0; dphi = 2*(x*vx + y*vy)
    if a is None:                       # naive index-1: phi_ddot = 0
        lam = (v2 - g*y)/r2
    else:                               # Baumgarte: phi_dd + 2a phi_d + b^2 phi = 0
        lam = (2*(v2 - g*y) + 2*a*dphi + b*b*phi)/(2*r2)
    return [vx, vy, -lam*x, -lam*y - g]

s0 = [1.0, 0.0, 0.0, 0.0]; T = 20.0
viol = lambda S: np.abs(S[0]**2 + S[1]**2 - 1)

print("=== reproduce Lab 2.6 (RK45, rtol=1e-7, atol=1e-6) ===")
for lab, ab in [("naive", (None, None)), ("Baumgarte a=b=5", (5., 5.))]:
    sol = solve_ivp(rhs, (0, T), s0, method='RK45', rtol=1e-7, atol=1e-6, args=ab)
    print(f"{lab:18s} viol(t=20)={viol(sol.y[:, -1]):.2e}  max viol={viol(sol.y).max():.2e}  nfev={sol.nfev}")

print("\n=== 2.3.B sweep a=b (RK45 vs Radau, same tol) ===")
print(" a=b    viol(20)   maxviol    nfev_RK45  nfev_Radau  fast-root |s|")
for ab in [0.5, 1, 5, 20, 100, 500, 2000]:
    s1 = solve_ivp(rhs, (0, T), s0, method='RK45', rtol=1e-7, atol=1e-6, args=(ab, ab))
    s2 = solve_ivp(rhs, (0, T), s0, method='Radau', rtol=1e-7, atol=1e-6, args=(ab, ab))
    roots = np.roots([1, 2*ab, ab*ab])
    print(f"{ab:6g}  {viol(s1.y[:, -1]):.2e}  {viol(s1.y).max():.2e}  {s1.nfev:9d}  {s2.nfev:9d}   {max(abs(roots)):.1f}")

print("\n=== 2.3.B critical damping a=b vs underdamped a<b (b=20) ===")
for a in [2, 5, 20, 80]:
    s1 = solve_ivp(rhs, (0, T), s0, method='RK45', rtol=1e-7, atol=1e-6, args=(a, 20.))
    print(f"a={a:3g} b=20  viol(20)={viol(s1.y[:, -1]):.2e}  maxviol={viol(s1.y).max():.2e}  nfev={s1.nfev}")

print("\n=== 2.3.C fixed-step RK4 dt=0.01 to t=20: naive / Baumgarte(5,5) / projection ===")
def rk4(f, s, dt, *args):
    k1 = np.array(f(0, s, *args)); k2 = np.array(f(0, s+0.5*dt*k1, *args))
    k3 = np.array(f(0, s+0.5*dt*k2, *args)); k4 = np.array(f(0, s+dt*k3, *args))
    return s + dt/6*(k1+2*k2+2*k3+k4)
def project(s):
    x, y, vx, vy = s; r = np.hypot(x, y); x, y = x/r, y/r
    vr = vx*x + vy*y                    # remove radial velocity -> phi_dot = 0
    return np.array([x, y, vx - vr*x, vy - vr*y])
energy = lambda s: 0.5*(s[2]**2 + s[3]**2) + g*s[1]
for dt in [0.01, 0.002]:
    n = int(round(T/dt))
    for name in ["naive", "baumgarte", "projection"]:
        s = np.array(s0); vmax = 0; Emax = 0
        for i in range(n):
            if name == "naive": s = rk4(rhs, s, dt)
            elif name == "baumgarte": s = rk4(rhs, s, dt, 5., 5.)
            else: s = project(rk4(rhs, s, dt))
            vmax = max(vmax, viol(s)); Emax = max(Emax, abs(energy(s) - energy(np.array(s0))))
        print(f"dt={dt}: {name:10s} viol(20)={viol(s):.2e}  maxviol={vmax:.2e}  max|E-E0|={Emax:.2e}  rhs evals={4*n}")

print("\n=== 2.3.B extra: fixed-step RK4 dt=0.01, Baumgarte a=b large (explicit stability limit) ===")
# double root s=-a ; RK4 real-axis stability |s dt| <= 2.785
for ab in [100, 250, 270, 300, 400]:
    s = np.array(s0); n = 2000; blown = False
    with np.errstate(all='ignore'):
        for i in range(n):
            s = rk4(rhs, s, 0.01, float(ab), float(ab))
            if not np.all(np.isfinite(s)) or viol(s) > 1e3: blown = True; break
    print(f"a=b={ab:4d}  |s|dt={ab*0.01:.2f}  ->  {'BLOW-UP at step %d' % i if blown else 'stable, viol(20)=%.2e' % viol(s)}")
