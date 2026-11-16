"""Exercise 10.7.4: cost of the minimal DFT plan in Estimate Box 10.11 units (one 5-atom SCF = 1 unit;
cost ~ N^3 with k-points scaled as 1/volume, so an N-atom cell costs (N/5)^3 * (5/N) = (N/5)^2 units)."""
unit = lambda N: (N / 5) ** 2
plan = [("phonons at a = 3.944 A, 2x2x2, 0.01 A (gives G, X, M, R)", 3, 40),
        ("same, 0.02 A (displacement check, Lab 10.13)", 3, 40),
        ("rigid a0a0c- tilt scan, 10-atom I4/mcm cell, 5 angles", 5, 10),
        ("I4/mcm relaxation of cell + O(8h) x, 10-atom cell, ~15 ionic steps", 15, 10)]
tot = 0; tot_min = 0
for lab, n, N in plan:
    c = n * unit(N); tot += c
    if '0.02' not in lab: tot_min += c
    print(f"{lab:72s}: {n:2d} x {unit(N):4.0f} = {c:5.0f} units")
print(f"total {tot:.0f} units (minimal plan without the 0.02 A check: {tot_min:.0f})")
lo, hi = 9 + 41 * 64, 9 + 47 * 64
print(f"whole Lab 10.12 by DFT: {lo:,}-{hi:,} units; plan / whole = {tot_min/hi:.1%}-{tot/lo:.1%}")
for t in (tot_min, tot):
    print(f"{t:.0f} units x 78 s = {t*78/3600:.1f} CPU h (PySCF minimal-basis benchmark; a lower bound)")
print(f"whole Lab by DFT: {lo*78/3600:.0f}-{hi*78/3600:.0f} CPU h")
