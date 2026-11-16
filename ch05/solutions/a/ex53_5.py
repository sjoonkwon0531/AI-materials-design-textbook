"""Ex 5.3.5: ZGB model with CO desorption. Each trial: pick a site; if it holds CO, desorb it with
probability p_d (trial ends); otherwise (or if not desorbed) do the usual ZGB adsorption attempt.
L = 64, 1000 warm-up + 1000 measured MCS, seed 1, y = 0.45 ... 0.70 step 0.01."""
import time
import numpy as np
from numba import njit

@njit(cache=True)
def zgb_des(L, y, pd, n_warm, n_meas, seed):
    np.random.seed(seed)
    s = np.zeros((L, L), dtype=np.int8)
    di = np.array([1, -1, 0, 0]); dj = np.array([0, 0, 1, -1])
    nCO = 0; nO = 0; sumCO = 0.0; sumO = 0.0; nprod = 0
    for mcs in range(n_warm + n_meas):
        for trial in range(L * L):
            i = np.random.randint(L); j = np.random.randint(L)
            if s[i, j] == 1 and np.random.random() < pd:     # CO desorption
                s[i, j] = 0; nCO -= 1
                continue
            if np.random.random() < y:
                if s[i, j] != 0:
                    continue
                d0 = np.random.randint(4); reacted = False
                for k in range(4):
                    d = (d0 + k) % 4
                    ii = (i + di[d]) % L; jj = (j + dj[d]) % L
                    if s[ii, jj] == 2:
                        s[ii, jj] = 0; nO -= 1; reacted = True
                        if mcs >= n_warm: nprod += 1
                        break
                if not reacted:
                    s[i, j] = 1; nCO += 1
            else:
                d = np.random.randint(4)
                i2 = (i + di[d]) % L; j2 = (j + dj[d]) % L
                if s[i, j] != 0 or s[i2, j2] != 0:
                    continue
                s[i, j] = 2; s[i2, j2] = 2; nO += 2
                for (pi, pj) in ((i, j), (i2, j2)):
                    d0 = np.random.randint(4)
                    for k in range(4):
                        dd = (d0 + k) % 4
                        ii = (pi + di[dd]) % L; jj = (pj + dj[dd]) % L
                        if s[ii, jj] == 1:
                            s[ii, jj] = 0; s[pi, pj] = 0; nCO -= 1; nO -= 1
                            if mcs >= n_warm: nprod += 1
                            break
        if mcs >= n_warm:
            sumCO += nCO; sumO += nO
    N2 = L * L
    return sumCO / n_meas / N2, sumO / n_meas / N2, nprod / n_meas / N2

t0 = time.time()
ys = np.round(np.arange(0.45, 0.7001, 0.01), 2)
res = {}
for pd in (0.0, 0.02, 0.1):
    rows = np.array([(y, *zgb_des(64, y, pd, 1000, 1000, 1)) for y in ys])
    res[pd] = rows
    print(f"=== p_d = {pd} ===")
    print("  y     theta_CO  theta_O  R_CO2")
    for r in rows:
        print(f"  {r[0]:.2f}  {r[1]:.4f}    {r[2]:.4f}   {r[3]:.4f}")
    jump = np.argmax(np.abs(np.diff(rows[:, 1])))
    imax = np.argmax(rows[:, 3])
    print(f"  max R_CO2 = {rows[imax,3]:.4f} at y = {rows[imax,0]:.2f};  largest single-step jump of theta_CO: "
          f"{rows[jump,1]:.3f} -> {rows[jump+1,1]:.3f} between y = {rows[jump,0]:.2f} and {rows[jump+1,0]:.2f}")
print(f"total {time.time()-t0:.0f} s")
