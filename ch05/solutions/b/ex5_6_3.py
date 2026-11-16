"""Ex 5.6.3: statistical error of the HMM micro estimator vs number of walkers; bias of short runs."""
import sys, time, numpy as np
sys.path.insert(0, "../ch05/labs")
import importlib; H = importlib.import_module("56_hmm")
exact = 0.5 + 0.5; exact = 1.0*H.HARM
t0 = time.time()
Ns = [500, 1000, 2000, 4000, 8000]; sd = []
for N in Ns:
    D = np.array([H.micro_D(0.5, nwalk=N, seed=s)[0] for s in range(10)])
    sd.append(D.std(ddof=1))
    print(f"N={N:5d}: mean={D.mean():.4f}, sample std={D.std(ddof=1):.4f}, std*sqrt(N)={D.std(ddof=1)*np.sqrt(N):.3f}")
p = np.polyfit(np.log(Ns), np.log(sd), 1)
print(f"log-log slope of std vs N = {p[0]:.3f}  (expected -0.5)")
# nsteps=1000 with the default nrec=80 is NOT safe: record index s//every-1 reaches 1000//12-1 = 82 > 79
print(f"nsteps=1000, nrec=80: every={1000//80}, largest record index={1000//(1000//80)-1} (array length 80)")
for nrec in (40, 50):
    D = np.array([H.micro_D(0.5, nwalk=4000, nsteps=1000, nrec=nrec, seed=s)[0] for s in range(10)])
    _, t, Dt, Lrun = H.micro_D(0.5, nwalk=4000, nsteps=1000, nrec=nrec, seed=0)
    print(f"nsteps=1000, nrec={nrec}: mean D = {D.mean():.4f} +/- {D.std(ddof=1)/np.sqrt(10):.4f} (sem), "
          f"exact {exact:.4f}, bias {D.mean()/exact-1:+.1%}; run length = {Lrun:.2f} eps^2/<D>")
D8 = np.array([H.micro_D(0.5, nwalk=4000, seed=s)[0] for s in range(10)])
print(f"reference nsteps=8000 (nrec=80): mean D = {D8.mean():.4f} +/- {D8.std(ddof=1)/np.sqrt(10):.4f} (sem)")
print(f"wall {time.time()-t0:.0f} s")
