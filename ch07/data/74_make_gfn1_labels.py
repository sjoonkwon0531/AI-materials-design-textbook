"""Cheap 'pretraining' labels for Lab 74: GFN1-xTB (tblite) single points on the SAME geometries as
every 3rd GFN2-xTB 600 K frame and every 2nd 1000 K frame (1,000 + 500 = 1,500 frames;ethanol_xtb_600K.npz, ethanol_xtb_1000K.npz).
GFN1-xTB plays the role of a lower level of theory; GFN2-xTB at 300 K is the 'expensive' target.
Output: 74_ethanol_gfn1_600_1000K.npz with R (1500,9,3), Z, E1 (eV), F1 (eV/A), T (per frame)."""
import numpy as np, time
from ase import Atoms
from tblite.ase import TBLite
t0 = time.time()
Rs, Es, Fs, Ts = [], [], [], []
for T in (600, 1000):
    d = np.load(f"ethanol_xtb_{T}K.npz")
    a = Atoms(numbers=d["Z"], positions=d["R"][0])
    a.calc = TBLite(method="GFN1-xTB", verbosity=0)
    idx = np.arange(0, len(d["R"]), 3 if T == 600 else 2)
    for R in d["R"][idx]:
        a.set_positions(R)
        Es.append(a.get_potential_energy()); Fs.append(a.get_forces().copy()); Rs.append(R); Ts.append(T)
    Z = d["Z"]
E1, F1 = np.array(Es), np.array(Fs)
np.savez("74_ethanol_gfn1_600_1000K.npz", R=np.array(Rs), Z=Z, E1=E1, F1=F1, T=np.array(Ts))
# compare with GFN2 on the same frames
E2 = np.concatenate([np.load(f"ethanol_xtb_{T}K.npz")["E"][::(3 if T == 600 else 2)] for T in (600, 1000)])
F2 = np.concatenate([np.load(f"ethanol_xtb_{T}K.npz")["F"][::(3 if T == 600 else 2)] for T in (600, 1000)])
dE = (E1 - E1.mean()) - (E2 - E2.mean())
print(f"frames {len(E1)}; GFN1 vs GFN2 on same geometries: energy (mean-shifted) MAE {np.abs(dE).mean()*1e3:.1f} meV, "
      f"force MAE {np.abs(F1-F2).mean()*1e3:.1f} meV/A, force corr {np.corrcoef(F1.ravel(),F2.ravel())[0,1]:.4f}; "
      f"mean offset E1-E2 = {(E1-E2).mean():.2f} eV; time {time.time()-t0:.1f} s")
