# Ex 4.6.6: one workflow (EOS / BFGS relax / Langevin MD), three calculators: EMT, user LJ, user ML(KRR on LJ)
import numpy as np, time, warnings; warnings.filterwarnings("ignore")
from ase import Atoms, units
from ase.build import bulk
from ase.cluster.cubic import FaceCenteredCubic
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.neighborlist import neighbor_list
from ase.eos import EquationOfState
from ase.optimize import BFGS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

class PairCalculator(Calculator):
    """generic pair-potential ASE calculator: subclasses supply u(x), f(x)=-du/dx in reduced x=r/sigma"""
    implemented_properties = ['energy', 'forces']
    def __init__(self, sigma=2.33, epsilon=0.40, xc=2.5, **kw):
        super().__init__(**kw); self.sigma, self.eps, self.xc = sigma, epsilon, xc
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        i, j, d, D = neighbor_list('ijdD', self.atoms, self.xc*self.sigma)   # every pair appears twice
        x = d/self.sigma; ushift = self.u(np.array([self.xc]))[0]
        E = 0.5*self.eps*np.sum(self.u(x) - ushift)
        fij = (self.eps/self.sigma)*self.f(x)                                 # magnitude, + = repulsive
        F = np.zeros((len(self.atoms),3)); np.add.at(F, i, -(fij/d)[:,None]*D)  # D = r_j - r_i
        self.results = {'energy': E, 'forces': F}
class LJCalc(PairCalculator):                          # Lab 4.4 force law
    def u(self, x): return 4*(x**-12 - x**-6)
    def f(self, x): return 48*x**-13 - 24*x**-7
class KRRCalc(PairCalculator):                         # Lab 4.5 KRR surrogate, trained on LJ "reference" data
    def __init__(self, n=160, gamma=2.0, lam=1e-8, seed=2, **kw):
        super().__init__(**kw); g = np.random.default_rng(seed)
        self.r = g.uniform(0.92, 2.5, n); y = 4*(self.r**-12 - self.r**-6); self.g = gamma
        K = np.exp(-gamma*(self.r[:,None]-self.r[None,:])**2); self.a = np.linalg.solve(K + lam*np.eye(n), y)
    def u(self, x): return np.exp(-self.g*(x[:,None]-self.r[None,:])**2) @ self.a
    def f(self, x):
        dd = x[:,None]-self.r[None,:]; return -((np.exp(-self.g*dd**2)*(-2*self.g*dd)) @ self.a)

# ---------------- calculator-agnostic workflows ----------------
def wf_eos(make_calc, a_guess=3.6):
    cu = bulk('Cu','fcc',a=a_guess,cubic=True); c0 = cu.get_cell(); V,E = [],[]
    for s in np.linspace(0.94,1.06,11):
        c = cu.copy(); c.set_cell(c0*s, scale_atoms=True); c.calc = make_calc()
        V.append(c.get_volume()); E.append(c.get_potential_energy())
    v0,e0,B = EquationOfState(V,E).fit(); return v0**(1/3), B/units.kJ*1e24
def wf_relax(make_calc):
    cl = Atoms(FaceCenteredCubic('Cu', surfaces=[(1,0,0),(1,1,1)], layers=[1,1], latticeconstant=3.6))
    cl.rattle(0.1, seed=1); cl.calc = make_calc(); e0 = cl.get_potential_energy()
    opt = BFGS(cl, logfile=None); opt.run(fmax=0.02, steps=300); return e0 - cl.get_potential_energy(), opt.nsteps
def wf_md(make_calc, a, nsteps=500):
    sc = bulk('Cu','fcc',a=a).repeat((4,4,4)); sc.calc = make_calc()
    MaxwellBoltzmannDistribution(sc, temperature_K=300, rng=np.random.RandomState(0))
    dyn = Langevin(sc, timestep=2*units.fs, temperature_K=300, friction=0.02, rng=np.random.RandomState(1))
    T=[]; dyn.attach(lambda: T.append(sc.get_temperature()), interval=10)
    t0 = time.perf_counter(); dyn.run(nsteps); dt = time.perf_counter()-t0
    return np.mean(T[len(T)//2:]), 1e3*dt/nsteps
calcs = {'EMT': EMT, 'LJ (user)': LJCalc, 'ML-KRR Lab(g=2)': KRRCalc,
         'ML-KRR tuned': lambda: KRRCalc(gamma=20.0, lam=1e-10)}
print(f"{'calculator':15s} {'a0 (A)':>8s} {'B (GPa)':>8s} {'dE_relax(eV)':>12s} {'BFGS steps':>10s} {'<T> (K)':>8s} {'ms/MD step':>10s}")
for name, mk in calcs.items():
    a0, B = wf_eos(mk); dE, ns = wf_relax(mk); T, ms = wf_md(mk, a0)
    print(f"{name:15s} {a0:8.4f} {B:8.1f} {dE:12.4f} {ns:10d} {T:8.1f} {ms:10.2f}")
# direct check: ML vs LJ forces on the same rattled configuration
sc = bulk('Cu','fcc',a=3.6).repeat((3,3,3)); sc.rattle(0.05, seed=3)
s1 = sc.copy(); s1.calc = LJCalc()
for tag, mk in [('Lab(g=2)', KRRCalc), ('tuned', lambda: KRRCalc(gamma=20.0, lam=1e-10))]:
    s2 = sc.copy(); s2.calc = mk(); k = s2.calc
    print(f"ML {tag} vs LJ on rattled Cu108: |dE|/atom = {abs(s1.get_potential_energy()-s2.get_potential_energy())/len(sc):.2e} eV, "
          f"force RMSE = {np.sqrt(np.mean((s1.get_forces()-s2.get_forces())**2)):.2e} eV/A, "
          f"pair-energy error at cutoff u_ML(2.5)-u_LJ(2.5) = {k.u(np.array([2.5]))[0]-4*(2.5**-12-2.5**-6):+.4f} eps")
