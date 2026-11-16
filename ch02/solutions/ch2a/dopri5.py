# Minimal Dormand-Prince 5(4) with FSAL, counters, and pluggable step controller
import numpy as np
C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
Am = [[], [1/5], [3/40, 9/40], [44/45, -56/15, 32/9],
      [19372/6561, -25360/2187, 64448/6561, -212/729],
      [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
      [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]]
B5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
B4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
E = B5 - B4
def step(f, t, y, h, k0):
    K = [k0]
    for i in range(1, 7):
        yi = y + h*sum(Am[i][j]*K[j] for j in range(i))
        K.append(f(t + C[i]*h, yi))
    ynew = y + h*sum(B5[i]*K[i] for i in range(6))      # = y + h*sum(A[6]) ; K[6]=f(t+h,ynew) (FSAL)
    err = h*sum(E[i]*K[i] for i in range(7))
    return ynew, err, K[6]
def integrate(f, t0, y0, T, rtol, atol, h0, controller="I", safety=0.9, fmin=0.2, fmax=10.0,
              kI=None, kP=None):
    """controller 'I': h*=s*err^(-1/5);  'PI': h*=s*err^(-(kI+kP))*err_prev^(kP) (Gustafsson)."""
    kI = 0.3/5 if kI is None else kI; kP = 0.4/5 if kP is None else kP
    t, y, h = t0, np.array(y0, float), h0
    k0 = f(t, y); nfev = 1; acc = rej = 0; hs = []; rej_y = []; err_prev = 1e-4; rejected_last = False
    while t < T - 1e-14:
        h = min(h, T - t)
        ynew, e, k6 = step(f, t, y, h, k0); nfev += 6
        sc = atol + rtol*np.maximum(np.abs(y), np.abs(ynew))
        err = max(np.sqrt(np.mean((e/sc)**2)), 1e-10)
        if controller == "I":
            fac = safety*err**(-1/5)
        else:
            fac = safety*err**(-(kI + kP))*err_prev**kP
        if err <= 1.0:
            t += h; y = ynew; k0 = k6; acc += 1; hs.append(h)
            if controller != "I": err_prev = err
            fac = min(fmax, max(fmin, fac))
            if rejected_last: fac = min(fac, 1.0)
            rejected_last = False
        else:
            rej += 1; rejected_last = True; rej_y.append(y.copy())
            fac = max(fmin, min(fac, 1.0) if controller == "I" else min(safety*err**(-1/5), 1.0))
        h *= fac
    return t, y, dict(nfev=nfev, accepted=acc, rejected=rej, hs=np.array(hs), rej_y=np.array(rej_y))
