"""Exercise 10.6.6: ghost-atom fraction [(L+2d)^3 - L^3]/L^3 for a cubic domain of side L."""
print("   d (A) |  L = 20 A   L = 50 A   L = 100 A")
for d in (6, 12, 24):
    print(f"  {d:5d}  | " + "  ".join(f"{((L+2*d)**3 - L**3)/L**3:9.3f}" for L in (20, 50, 100)))
# Si density: atoms in a domain and in its halo for L = 50 A
n = 8 / 5.43**3
for d in (6, 12, 24):
    L = 50
    print(f"L = 50 A, d = {d}: {n*L**3:,.0f} owned atoms, {n*((L+2*d)**3-L**3):,.0f} ghost atoms")
