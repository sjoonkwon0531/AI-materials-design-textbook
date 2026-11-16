# Ex 4.7.2: reproduce Lab 4.7 site energies (EMT) and fcc-top difference
from common_pt import *
print(f"E(H2, EMT) = {eH2:.4f} eV")
res = {}
for s in ['fcc','hcp','bridge','ontop']:
    E, sl = E_ads_site(s, return_atoms=True); res[s] = E
    d = sl.get_distances(len(sl)-1, range(len(sl)-1), mic=True)
    print(f"  {s:7s} E_ads = {E:+.4f} eV   H height above top layer = {sl.positions[-1,2]-sl.positions[:-1,2].max():.3f} A   min d(Pt-H) = {d.min():.3f} A")
print(f"E_ads(top) - E_ads(fcc) = {res['ontop']-res['fcc']:+.4f} eV;  hcp - fcc = {res['hcp']-res['fcc']:+.4f} eV")
