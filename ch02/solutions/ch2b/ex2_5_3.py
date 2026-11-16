# Ex 2.5.3: BTCS recursive matrix solution with a time-dependent source q = 100 exp(-eta tau) on 0.45<=r<=0.55
# Interpretation (stated in the solution): U = T/T0, T0 = 25 C at both walls and initially;
# dimensionless U_tau = U_rr + q(r,tau);  work with the excess V = U - 1 (V=0 on walls, V(.,0)=0).
import numpy as np
from scipy.optimize import brentq
T0, q0 = 25.0, 100.0

def setup(h, dtau):
    N = int(round(1/h)); r = np.linspace(0, 1, N+1)[1:-1]; n = N-1; lam = dtau/h**2
    A = (1+2*lam)*np.eye(n) - lam*(np.eye(n, k=1) + np.eye(n, k=-1)); D = np.linalg.inv(A)
    w = np.clip((0.05 - np.abs(r-0.5))/h + 0.5, 0, 1)   # cell-fraction weight: exact width 0.1 of the heated strip
    return r, D, w, lam

def recursive(h, dtau, eta, nsteps, form='implicit'):
    r, D, w, lam = setup(h, dtau); n = len(r)
    U0 = np.zeros(n); Dn = np.eye(n); acc = np.zeros(n); peak = []
    for N in range(1, nsteps+1):
        # implicit source:  U_N = D^N U0 + dtau sum_{m=1}^{N} D^{N-m+1} q_m
        # book form      :  U_N = D^N U0 + dtau sum_{m=0}^{N-1} D^{N-m-1} q_m   (source added explicitly)
        qm = q0*np.exp(-eta*N*dtau)*w if form == 'implicit' else q0*np.exp(-eta*(N-1)*dtau)*w
        acc = D@acc + dtau*(D@qm if form == 'implicit' else qm)    # running sum, same algebra as the closed form
        Dn = D@Dn
        peak.append((Dn@U0 + acc).max())
    return r, Dn@U0 + acc, np.array(peak)

# 1) closed form (explicit sum of matrix powers) == running recursion, small case
h, dtau = 0.05, 0.0025; r, D, w, lam = setup(h, dtau); n = len(r); nst = 40
U_loop = np.zeros(n)
for N in range(1, nst+1): U_loop = D@(U_loop + dtau*q0*np.exp(-1.0*N*dtau)*w)
U_sum = sum(np.linalg.matrix_power(D, nst-m+1)@(dtau*q0*np.exp(-1.0*m*dtau)*w) for m in range(1, nst+1))
print(f"[check] closed-form sum vs step loop, max diff = {np.abs(U_loop-U_sum).max():.2e};  rho(D) = {max(abs(np.linalg.eigvals(D))):.4f}")

# 2) eta = 0 (constant source): steady state vs analytic  V_max = q a (1/2 - a/2), a = 0.05
Vmax = q0*0.05*(0.5-0.025); print(f"\nanalytic steady excess V_max = {Vmax:.4f}  -> T_max = {T0*(1+Vmax):.3f} C")
for h, dtau in [(0.05, 0.0025), (0.02, 0.0004), (0.01, 1e-3)]:
    lam = dtau/h**2; nst = int(round(1.5/dtau))
    for form in ['implicit', 'book']:
        r, V, pk = recursive(h, dtau, 0.0, nst, form)
        print(f"h={h:5.3f} dtau={dtau:.0e} (lam={lam:4.1f}) {form:8s}: T_max(tau=1.5) = {T0*(1+V.max()):.3f} C")

# 3) eta > 0: transient peak, then decay back to T0 (no non-trivial steady state)
print("\neta>0 (h=0.01, dtau=1e-3, implicit source):")
for eta in [1.0, 3.0, 10.0]:
    r, V, pk = recursive(0.01, 1e-3, eta, 3000, 'implicit')
    k = pk.argmax()
    print(f"eta={eta:4.1f}: peak T_max = {T0*(1+pk[k]):.2f} C at tau = {(k+1)*1e-3:.3f};  T_max(tau=3) = {T0*(1+V.max()):.3f} C")

# 4) which eta would give a transient peak of 44.46 C under THIS interpretation (diagnostic only)
f = lambda eta: T0*(1+recursive(0.01, 1e-3, eta, 1500, 'implicit')[2].max()) - 44.46
eta_star = brentq(f, 10, 200, xtol=1e-3)
print(f"\n[diagnostic] transient peak = 44.46 C would need eta ~ {eta_star:.1f} (h=0.01, dtau=1e-3) under this interpretation")
