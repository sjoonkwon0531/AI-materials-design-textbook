"""Cross-check for rk45_scipy.jl: numpy re-implementation of the Julia DOPRI5 (SciPy-RK45-faithful)
logic, compared against scipy.integrate.solve_ivp(method='RK45') on Labs 2.3, 2.4, 2.6."""
import numpy as np
from scipy.integrate import solve_ivp
C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1.0])
A = [[], [1/5], [3/40, 9/40], [44/45, -56/15, 32/9],
     [19372/6561, -25360/2187, 64448/6561, -212/729],
     [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]]
B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40])
rms = lambda v: np.sqrt(np.sum(v*v)/v.size)
def rk45(f, t0, y0, T, rtol, atol):
    t = t0; y = np.array(y0, float); fy = f(t, y); nfev = 1
    sc = atol + np.abs(y)*rtol; d0 = rms(y/sc); d1 = rms(fy/sc)
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01*d0/d1
    h0 = min(h0, T-t0)
    f1 = f(t+h0, y+h0*fy); nfev += 1
    d2 = rms((f1-fy)/sc)/h0
    h1 = max(1e-6, h0*1e-3) if (d1 <= 1e-15 and d2 <= 1e-15) else (0.01/max(d1, d2))**(1/5)
    habs = min(100*h0, h1, T-t0)
    ts = [t]; ys = [y.copy()]; rej = 0
    while t < T:
        minstep = 10*abs(np.nextafter(t, np.inf)-t)
        habs = max(habs, minstep)
        rejected = False
        while True:
            tnew = min(t+habs, T); h = tnew-t; habs = abs(h)
            K = [fy]
            for s in range(1, 6):
                K.append(f(t+C[s]*h, y + h*sum(A[s][j]*K[j] for j in range(s))))
            ynew = y + h*sum(B[j]*K[j] for j in range(6))
            fnew = f(t+h, ynew); K.append(fnew); nfev += 6
            sc = atol + np.maximum(np.abs(y), np.abs(ynew))*rtol
            err = rms(h*sum(E[j]*K[j] for j in range(7))/sc)
            if err < 1:
                fac = 10.0 if err == 0 else min(10.0, 0.9*err**(-0.2))
                if rejected: fac = min(1.0, fac)
                habs *= fac; break
            habs *= max(0.2, 0.9*err**(-0.2)); rejected = True; rej += 1
        t = tnew; y = ynew; fy = fnew; ts.append(t); ys.append(y.copy())
    return np.array(ts), np.array(ys), nfev, rej
# Lab 2.4 Kepler
kep = lambda t, Y: np.array([Y[2], Y[3], -Y[0]/(Y[0]**2+Y[1]**2)**1.5, -Y[1]/(Y[0]**2+Y[1]**2)**1.5])
y0 = [0.2, 0, 0, 3.0]
ts, ys, nf, rej = rk45(kep, 0, y0, 8*np.pi, 1e-8, 1e-11)
s = solve_ivp(kep, (0, 8*np.pi), y0, method='RK45', rtol=1e-8, atol=1e-11)
print("Kepler mine steps", len(ts)-1, "nfev", nf, "rej", rej, "| scipy", len(s.t)-1, s.nfev, "maxdiff", np.abs(ys[-1]-s.y[:, -1]).max())
dt = np.diff(ts); d = dt[2:-1]; print(" interior dt min %.4g max %.4g ratio %.0f" % (d.min(), d.max(), d.max()/d.min()))
r = np.hypot(ys[:, 0], ys[:, 1]); En = 0.5*(ys[:, 2]**2+ys[:, 3]**2)-1/r; print(" energy drift %.2e" % (En[-1]/En[0]-1))
# Lab 2.3 stiff
Am = np.array([[-1001.5, 0, 0], [1001.5, -0.5, 1.0], [0, 0.5, -1.0]])
ts, ys, nf, rej = rk45(lambda t, x: Am@x, 0, [1., 0, 0], 8.0, 1e-6, 1e-9)
s = solve_ivp(lambda t, x: Am@x, (0, 8), [1., 0, 0], method='RK45', rtol=1e-6, atol=1e-9)
print("stiff mine steps", len(ts)-1, "nfev", nf, "| scipy", len(s.t)-1, s.nfev)
# Lab 2.6 pendulum
g = 9.81
def mk(al, be):
    def f(t, u):
        x, y, vx, vy = u; r2 = x*x+y*y; v2 = vx*vx+vy*vy; phi = r2-1; dphi = 2*(x*vx+y*vy)
        lam = (v2-g*y+al*dphi+0.5*be**2*phi)/r2
        return np.array([vx, vy, -lam*x, -lam*y-g])
    return f
for ab in (0, 5):
    ts, ys, nf, rej = rk45(mk(ab, ab), 0, [1., 0, 0, 0], 20.0, 1e-7, 1e-6)
    s = solve_ivp(mk(ab, ab), (0, 20), [1., 0, 0, 0], method='RK45', rtol=1e-7, atol=1e-6)
    v = np.abs(ys[:, 0]**2+ys[:, 1]**2-1)
    print("pend ab=", ab, "mine nfev", nf, "|phi|(20)=%.2e max(steps)=%.2e" % (v[-1], v.max()), "| scipy nfev", s.nfev, "|phi|(20)=%.2e" % abs(s.y[0, -1]**2+s.y[1, -1]**2-1))
