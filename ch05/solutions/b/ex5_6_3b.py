"""Ex 5.6.3 (cont.): exact expectation of the micro estimator (no sampling noise), from the
first and second displacement moments of the discrete-time ring walk, plus a 100-seed MC check."""
import sys, numpy as np
sys.path.insert(0, "../ch05/labs")
import importlib; H = importlib.import_module("56_hmm")

def exact_estimator(x0=0.5, m=16, nsteps=8000, nrec=80):
    h = H.EPS/m; y = (np.arange(m)+0.5)/m
    w = H.A(x0)*H.a(y)/h**2; dtau = 0.5/w.max()
    pr = w*dtau; pl = np.roll(w, 1)*dtau; ps = 1-pr-pl
    P = np.full(m, 1.0/m); M1 = np.zeros(m); M2 = np.zeros(m)
    every = nsteps//nrec; msd = []
    for s in range(1, nsteps+1):
        # move right: site i -> i+1 (pos+1); left: i -> i-1 (pos-1)
        Pn = ps*P + np.roll(pr*P, 1) + np.roll(pl*P, -1)
        M1n = ps*M1 + np.roll(pr*(M1+P), 1) + np.roll(pl*(M1-P), -1)
        M2n = ps*M2 + np.roll(pr*(M2+2*M1+P), 1) + np.roll(pl*(M2-2*M1+P), -1)
        P, M1, M2 = Pn, M1n, M2n
        if s % every == 0 and len(msd) < nrec: msd.append(M2.sum())
    msd = np.array(msd)*h**2; t = dtau*every*np.arange(1, len(msd)+1)
    half = len(msd)//2
    return np.polyfit(t[half:], msd[half:], 1)[0]/2, msd[-1]/(2*t[-1]), msd[0]/(2*t[0])

ex = H.HARM; yb = (np.arange(16)+0.5)/16; latt = 1/np.mean(1/H.a(yb))
print(f"continuum harmonic {ex:.4f}, lattice harmonic (16 bonds) {latt:.4f}")
for ns, nr in ((8000, 80), (1000, 40), (1000, 50), (2000, 40), (4000, 40)):
    D, Dlast, Dfirst = exact_estimator(nsteps=ns, nrec=nr)
    print(f"nsteps={ns}, nrec={nr}: E[slope estimate] = {D:.4f} (bias vs lattice {D/latt-1:+.2%}, vs continuum {D/ex-1:+.2%}); "
          f"MSD/2t first {Dfirst:.4f}, last {Dlast:.4f}")
Ds = np.array([H.micro_D(0.5, nwalk=4000, nsteps=1000, nrec=40, seed=s)[0] for s in range(100, 200)])
print(f"MC, 100 seeds, nsteps=1000, nrec=40: mean = {Ds.mean():.4f} +/- {Ds.std(ddof=1)/10:.4f}")
# where does the slope estimator itself become biased? (exact expectation)
for ns, nr in ((40, 20), (80, 20), (160, 20), (320, 20)):
    D, Dlast, Dfirst = exact_estimator(nsteps=ns, nrec=nr)
    h = H.EPS/16; w = H.a((np.arange(16)+0.5)/16)/h**2; dtau = 0.5/w.max()
    print(f"nsteps={ns:4d} (run {ns*dtau/H.EPS**2:.3f} eps^2/<D>): E[slope]={D:.4f} ({D/latt-1:+.1%}), E[MSD/2t last]={Dlast:.4f} ({Dlast/latt-1:+.1%})")
# the 10-seed numbers asked in the exercise, both estimators
res = [H.micro_D(0.5, nwalk=4000, nsteps=1000, nrec=40, seed=s) for s in range(10)]
sl = np.array([r[0] for r in res]); last = np.array([r[2][-1] for r in res])
print(f"10 seeds, nsteps=1000, nrec=40: slope mean {sl.mean():.4f}+/-{sl.std(ddof=1)/np.sqrt(10):.4f}; "
      f"MSD/2t(last) mean {last.mean():.4f}+/-{last.std(ddof=1)/np.sqrt(10):.4f}")
