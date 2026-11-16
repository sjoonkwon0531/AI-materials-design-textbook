"""Ex 1.9.E: 3D Ising (simple cubic, 6 neighbours) Metropolis; estimate Tc at L=16
(chi' peak) and from Binder-cumulant crossing of L=8 and L=16. Reference Tc ~ 4.5115."""
import numpy as np, time
from ising_lib import seed, run3d
seed(5); rng = np.random.default_rng(5)
Ts = np.round(np.arange(4.70, 4.3999, -0.025), 3)       # cool down, reuse configuration
out = {}
t0 = time.time()
for L, neq, nme in [(8, 5000, 100000), (16, 5000, 50000)]:
    s = rng.choice(np.array([-1, 1], dtype=np.int64), size=(L, L, L)); rows = []
    for T in Ts:
        m = run3d(s, 1/T, neq, nme); a = np.abs(m); m2 = np.mean(m**2)
        rows.append((T, a.mean(), (1/T)*L**3*(m2 - a.mean()**2), 1 - np.mean(m**4)/(3*m2**2)))
    out[L] = np.array(rows[::-1]); print(f"L={L} done {time.time()-t0:.0f}s", flush=True)
for L in out:
    print(f"L={L}:"); [print(f"   T={r[0]:.3f} <|m|>={r[1]:.4f} chi'={r[2]:8.3f} U4={r[3]:.4f}") for r in out[L]]
r = out[16]; i = np.argmax(r[:, 2]); sl = slice(max(i-2, 0), i+3)
c = np.polyfit(r[sl, 0], r[sl, 2], 2); Tpk = -c[1]/(2*c[0])
print(f"L=16 chi' peak: T*={Tpk:.3f} (grid max at {r[i,0]:.3f})")
d = out[8][:, 3] - out[16][:, 3]; T = out[8][:, 0]
k = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
for kk in k:
    Tx = T[kk] - d[kk]*(T[kk+1]-T[kk])/(d[kk+1]-d[kk]); print(f"Binder crossing L=8/16: Tc ~ {Tx:.3f}, U* ~ {np.interp(Tx, T, out[16][:,3]):.3f}")
print(f"reference 4.5115; Metropolis L=16 sweeps total = {len(Ts)*(55000)}")
