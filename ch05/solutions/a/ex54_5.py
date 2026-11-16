"""Ex 5.4.5: (a) IBI damping alpha = 0.5, 1.0 (Lab), 1.5, nine iterations each, same seeds as the Lab;
(b) IBI from scratch at rho = 0.20 (target from fine MD at rho = 0.20, as in Lab step 5) vs the rho = 0.30 potential."""
import time
import numpy as np
from ibi_chain import get_chain, ibi_iterations, ibi, NMOL, T, DT, L

t0 = time.time()
c = get_chain()
g_t = c["g_t"]; r = ibi.RGRID
print("=== (a) g RMSE per iteration (noise floor 0.0194 from the Lab) ===")
runs = {1.0: c["errs"]}
Ps = {1.0: c["Ps"]}
for alpha in (0.5, 1.5):
    out = ibi_iterations(g_t, L, alpha, 9, 100, verbose=False)
    runs[alpha] = out["errs"]; Ps[alpha] = out["Ps"]
    if alpha == 1.5:
        V15 = out["V"]
for alpha in (0.5, 1.0, 1.5):
    print(f"  alpha={alpha}: RMSE " + " ".join(f"{e:.4f}" for e in runs[alpha]))
    print(f"  alpha={alpha}: P_CG " + " ".join(f"{p:.3f}" for p in Ps[alpha]))
print(f"  ({time.time()-t0:.0f} s)", flush=True)

print("=== (b) IBI at rho = 0.20 from its own fine-MD target ===")
L2 = np.sqrt(NMOL / 0.20)
h2, n2, P2f, _ = ibi.run_fine(NMOL, L2, 10000, 50000, DT, 1.0, 2, 40)
g2 = ibi.gofr(h2, n2, NMOL, L2)
print(f"  fine MD rho=0.20: P_fine = {P2f:.4f}; first peak g = {g2.max():.3f} at r = {r[np.argmax(g2)]:.3f} "
      f"(rho=0.30: {g_t.max():.3f} at {r[np.argmax(g_t)]:.3f})")
out2 = ibi_iterations(g2, L2, 1.0, 9, 700, verbose=False)
print("  rho=0.20 RMSE " + " ".join(f"{e:.4f}" for e in out2["errs"]))
print("  rho=0.20 P_CG " + " ".join(f"{p:.4f}" for p in out2["Ps"]) + f"  (P_fine = {P2f:.4f})")
# noise floor at rho = 0.20
hc, nf, _, _ = ibi.run_cg(NMOL, L2, out2["Ft"], 5000, 30000, DT, 1.0, 999, 15, 2.0)
print(f"  noise floor rho=0.20 (independent seed, same V): {ibi.gerr(ibi.gofr(hc, nf, NMOL, L2), out2['gs'][-1]):.4f}")
V3, V2 = c["V"], out2["V"]          # potentials used in the last IBI iteration at each density
def well(V):
    k = np.argmin(V); return V[k], r[k]
for name, V in (("rho=0.30", V3), ("rho=0.20", V2)):
    vm, rm = well(V)
    i23 = np.argmin(np.abs(r - 2.3)); i15 = np.argmin(np.abs(r - 1.2))
    print(f"  {name}: min V = {vm:.4f} at r = {rm:.2f};  V(1.2) = {V[i15]:.3f};  V(2.3) = {V[i23]:.4f};  "
          f"int_1^3.5 V dr = {np.sum(V[(r>1)&(r<3.5)])*ibi.DR:.4f}")
d = (V2 - V3)[(r > 1.0) & (r < 3.5)]
print(f"  V_0.20 - V_0.30 on 1<r<3.5: min {d.min():.4f}, max {d.max():.4f}")
# the rho=0.30 potential used at rho=0.20 (Lab step 5 numbers are 0.1054 / 0.2260) vs its own IBI result
print(f"  at rho=0.20: own IBI RMSE (last) = {out2['errs'][-1]:.4f}, P = {out2['Ps'][-1]:.4f}; Lab transferred rho=0.30 IBI: RMSE 0.1054, P 0.2260")
# and the rho=0.20 potential used at rho=0.30
hc, nf, P, _ = ibi.run_cg(NMOL, L, out2["Ft"], 5000, 30000, DT, 1.0, 888, 15, 2.0)
print(f"  rho=0.20 potential used at rho=0.30: g RMSE = {ibi.gerr(ibi.gofr(hc, nf, NMOL, L), g_t):.4f}, P = {P:.4f} (P_fine 0.2843)")
print(f"total {time.time()-t0:.0f} s")
