"""Ch.3 exercise solutions: all numerical results (3.1.B, 3.2.B, 3.2.C, 3.3.C, 3.5.C, 3.6.A, 3.6.B).
Run: python3 sol_ch3_numbers.py
"""
from math import comb, log10

print("=== 3.1.B ternary combinatorics ===")
n_comp = comb(80, 3)
n_total = n_comp * 100 * 200
gnome = 381_000  # Merchant et al. 2023 (book Sec. 3.1: ~381,000 stable)
print(f"C(80,3)            = {n_comp}")
print(f"candidates         = {n_total:.3e}  (log10 = {log10(n_total):.2f})")
print(f"GNoME fraction     = {gnome/n_total:.2e}  (= 1/{n_total/gnome:.0f})")
print(f"Box 3.1 quaternary C(80,4) = {comb(80,4)}  ratio quat/tern compositions = {comb(80,4)/n_comp:.1f}")

print("\n=== 3.2.B Amdahl ===")
def S(fracs, speed):
    return 1.0 / sum(f / s for f, s in zip(fracs, speed))
fr = (0.40, 0.35, 0.25)
print(f"(1) design x10          : S = {S(fr,(10,1,1)):.4f}")
print(f"(2) design x inf        : S = {1/(1-0.40):.4f}")
print(f"(3) make,test x5 each   : S = {S(fr,(1,5,5)):.4f}")
print(f"    make,test x inf     : S = {1/0.40:.4f}")
print(f"    design x10 + make,test x5 : S = {S(fr,(10,5,5)):.4f}")

print("\n=== 3.2.C budget illustration (Box 3.2 split 20/50/30) ===")
fr2 = (0.20, 0.50, 0.30)
base = S(fr2, (1e9, 1, 1))  # GNoME-like: design already ~instant
print(f"design instant only           : S = {base:.4f}")
for lab, sp in [("(i)  better AI (design stays instant)", (1e9, 1, 1)),
                ("(ii) make x5                    ", (1e9, 5, 1)),
                ("(iii) test x5                   ", (1e9, 1, 5)),
                ("(ii)+(iii) make x5, test x5     ", (1e9, 5, 5))]:
    print(f"{lab}: S = {S(fr2, sp):.4f}")
# marginal: equal 2x on make vs test
print(f"make x2 : S = {S(fr2,(1e9,2,1)):.4f} ; test x2 : S = {S(fr2,(1e9,1,2)):.4f}")
# Note on (i): a better AI that raises hit-rate h -> fewer cycles; cycles needed ~ 1/h
# (i) better AI acts through hit-rate (fewer cycles), not through Amdahl stage speed; discussed qualitatively.

print("\n=== 3.3.C connections ===")
for N, M in [(10, 22), (100, 22)]:
    print(f"N={N}, M={M}: N*M = {N*M}, N+M = {N+M}, ratio = {N*M/(N+M):.1f}")
