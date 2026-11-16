"""Ex 5.6.1 (ONIOM arithmetic), 5.6.2(c) (AdResS drift force), 5.6.4 (HMM vs fine cost)."""
# 5.6.1
Eh1, El1, Elt1 = -152.40, -150.10, -612.85
Eh2, El2, Elt2 = -151.90, -149.30, -612.30
On1 = Eh1 + Elt1 - El1; On2 = Eh2 + Elt2 - El2
print(f"5.6.1: E_ONIOM(R)={On1:.2f} eV, E_ONIOM(P)={On2:.2f} eV, dE_ONIOM={On2-On1:+.2f} eV, dE_low={Elt2-Elt1:+.2f} eV")
print(f"       correction = d(E_high(I)-E_low(I)) = {(Eh2-El2)-(Eh1-El1):+.2f} eV; dE_high(I)={Eh2-Eh1:+.2f}, dE_low(I)={El2-El1:+.2f}, env.(low) = {(Elt2-El2)-(Elt1-El1):+.2f}")
# 5.6.2(c)
dU = 2.0; width = 1.0; gradw = -1.0/width      # w: 1 -> 0 going from AT to CG over 1 nm
F = -dU*gradw
import math
print(f"5.6.2: drift force = -(U_AT-U_CG) grad w = {F:+.1f} kBT/nm (points toward the CG side); "
      f"integrated step {F*width:.1f} kBT -> equilibrium density ratio exp(2) = {math.exp(2):.2f}")
print(f"5.6.3 note: ring relaxation t_r ~ eps^2/(4 pi^2 * 0.436) = {1/(4*math.pi**2*0.436):.3f} eps^2")
# 5.6.4
L, eps, pts = 1e-3, 1e-6, 8
for e in (1e-6, 1e-7):
    fine = (L/e*pts)**3; hmm = 32**3*pts**3
    print(f"5.6.4: eps={e*1e6:.1f} um: fine={fine:.3e} points, HMM micro total={hmm:.3e}, ratio={fine/hmm:.3e}, macro cell/eps={L/32/e:.0f}")
