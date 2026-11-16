"""Ex 10.3.5 (c): ratio exp/model for Si C11, C12, C44 (relaxed ions, 1 % strain) and B; one scalar s cannot fix all."""
exp = (165.7, 63.9, 79.5); Bpbe = 88.5
mods = {'MACE': (95.7, 58.0, 29.2, 71.6), 'CHGNet': (105.8, 38.2, 37.3, 77.9),
        '7net-0': (152.1, 69.4, 57.8, 94.6), 'MatSim': (133.8, 67.1, 50.3, 89.3)}
for m, (c11, c12, c44, B) in mods.items():
    r = [exp[0] / c11, exp[1] / c12, exp[2] / c44]
    print(f"{m:7s}: s needed for C11 {r[0]:.2f}, C12 {r[1]:.2f}, C44 {r[2]:.2f}; for B vs PBE {Bpbe/B:.2f}; spread max/min {max(r)/min(r):.2f}")
