# Ex 2.2.G: velocity-Verlet == position Verlet == leapfrog (up to round-off)
import numpy as np
w2 = 1.0; a = lambda x: -w2*x            # harmonic; also test pendulum
for name, acc in [("harmonic", lambda x: -x), ("pendulum", lambda x: -np.sin(x))]:
    x0, v0, h, N = 1.0, 0.0, 0.1, 20000
    # velocity Verlet
    X = np.empty(N+1); X[0]=x0; x, v = x0, v0; an = acc(x)
    for n in range(N):
        x = x + v*h + 0.5*an*h*h; anew = acc(x); v = v + 0.5*(an+anew)*h; an = anew; X[n+1]=x
    # position Verlet, started with X[1] from the same first VV step
    P = np.empty(N+1); P[0]=x0; P[1]=X[1]
    for n in range(1, N):
        P[n+1] = 2*P[n] - P[n-1] + acc(P[n])*h*h
    # leapfrog: v_{1/2} = v0 + a0 h/2
    L = np.empty(N+1); L[0]=x0; x=x0; vh = v0 + 0.5*acc(x0)*h
    for n in range(N):
        x = x + vh*h; L[n+1]=x; vh = vh + acc(x)*h
    print(f"{name}: max|VV-posVerlet|={np.abs(X-P).max():.2e}, max|VV-leapfrog|={np.abs(X-L).max():.2e} over {N} steps")
