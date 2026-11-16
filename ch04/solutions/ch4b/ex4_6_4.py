# Ex 4.6.4: convergence of EMT surface energy and H adsorption energy of Pt(111) w.r.t. layers and vacuum
from common_pt import *
from ase.build import bulk
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a = 3.92                                                 # same lattice constant as fcc111('Pt') default
pb = bulk('Pt','fcc',a=a); pb.calc = EMT(); e_bulk = pb.get_potential_energy()/len(pb)
J = 16.0217662                                           # eV/A^2 -> J/m^2
def gamma(nl, vac, relaxed=False):
    s = fcc111('Pt', size=(1,1,nl), a=a, vacuum=vac); s.calc = EMT()
    A = np.linalg.norm(np.cross(s.cell[0], s.cell[1]))
    g = (s.get_potential_energy() - len(s)*e_bulk)/(2*A)          # unrelaxed, two identical faces
    if relaxed and nl >= 3:                                          # bottom 2 layers fixed, top relaxed (Lab protocol)
        cut = np.sort(np.unique(np.round(s.positions[:,2],2)))[1] + 0.1
        e0 = s.get_potential_energy(); s.set_constraint(FixAtoms(mask=s.positions[:,2] < cut))
        g += (relax(s, 0.001) - e0)/A                               # relaxation lowers only the top face
    return g*J
print(f"E_bulk(Pt, EMT, a={a}) = {e_bulk:.4f} eV/atom")
L = [3,4,5,6,7,8]; V = [6,9,12,15]
for n in [1,2]: print(f"  gamma(unrelaxed, {n} layer(s)) = {gamma(n,12.0):.5f} J/m^2")
gL = [gamma(n,12.0) for n in L]; gV = [gamma(4,v) for v in V]
gR = [gamma(n,12.0,True) for n in L]
EL = [E_ads_site('fcc', size=(2,2,n), vacuum=12.0) for n in [3,4,5,6]]
EV = [E_ads_site('fcc', size=(2,2,4), vacuum=float(v)) for v in V]
print("layers  gamma_unrel  gamma_top-relaxed (J/m^2)  E_ads(fcc, eV)   [vacuum 12 A]")
for i,n in enumerate(L): print(f"  {n}      {gL[i]:.5f}     {gR[i]:.5f}                 " + (f"{EL[i]:+.4f}" if n<=6 else "   -"))
print("vacuum  gamma(J/m^2)  E_ads(fcc, eV)   [4 layers]")
for v,g,e in zip(V,gV,EV): print(f"  {v:2d}     {g:.5f}      {e:+.4f}")
fig,ax=plt.subplots(1,2,figsize=(8,3))
ax[0].plot(L,gL,'o-'); ax[0].set_xlabel('layers'); ax[0].set_ylabel(r'$\gamma$ (J/m$^2$)')
ax2=ax[0].twinx(); ax2.plot([3,4,5,6],EL,'s--',color='C1'); ax2.set_ylabel('E_ads (eV)')
ax[1].plot(V,gV,'o-'); ax[1].set_xlabel('vacuum (A)'); ax3=ax[1].twinx(); ax3.plot(V,EV,'s--',color='C1')
plt.tight_layout(); plt.savefig('ex4_6_4_convergence.png',dpi=120)
