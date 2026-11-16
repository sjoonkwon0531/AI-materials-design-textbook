"""Ex 5.5.6(c): Keller-Dykhne product for the seed-1 CH map under sub-pixel refinement.
Both K(1,10) and K(10,1) are displacement-FEM upper bounds (in the matrix sense), so the product of
determinants approaches (k1 k2)^2 from above."""
import sys, time, numpy as np
sys.path.insert(0, "../../../ch05/labs")
import importlib; H = importlib.import_module("55_homog")
img = np.load("../../../ch05/labs/55_homog_imgs.npz")["m+0.00"]
t0 = time.time(); prev = None
for r in [int(a) for a in sys.argv[1:]] or (1, 2, 4):   # r=8 needs > 7 GB with spsolve
    cell = H.PeriodicCell(128, r)
    Ka = H.k_eff(cell, img, 1.0, 10.0); Kb = H.k_eff(cell, img, 10.0, 1.0)
    da, db = np.linalg.det(Ka), np.linalg.det(Kb); prod = np.sqrt(da*db)
    print(f"r={r}: K(1,10) xx,yy,xy = {Ka[0,0]:.5f},{Ka[1,1]:.5f},{Ka[0,1]:+.5f}; sqrt det = {np.sqrt(da):.5f} | "
          f"K(10,1) xx,yy = {Kb[0,0]:.5f},{Kb[1,1]:.5f}; sqrt det = {np.sqrt(db):.5f} | product = {prod:.5f}"
          + ("" if prev is None else f"  (excess ratio to previous {(prev-10)/(prod-10):.2f})") + f"  [{time.time()-t0:.0f} s]", flush=True)
    pred = 10.0*Ka/da                      # Keller: K(k2,k1) = k1 k2 K(k1,k2) / det K(k1,k2)
    print(f"      Keller prediction from K(1,10): K(10,1) xx,yy,xy = {pred[0,0]:.5f},{pred[1,1]:.5f},{pred[0,1]:+.5f}  "
          f"vs computed {Kb[0,0]:.5f},{Kb[1,1]:.5f},{Kb[0,1]:+.5f}")
    prev = prod
# also: Keller holds for the swapped *geometry* too: K(1,10) on img vs K(1,10) on ~img
cell = H.PeriodicCell(128, 1)
Kc = H.k_eff(cell, ~img, 1.0, 10.0); Kd = H.k_eff(cell, img, 10.0, 1.0)
print(f"check: K(10,1) on map == K(1,10) on swapped map: {np.allclose(Kc, Kd)}")
