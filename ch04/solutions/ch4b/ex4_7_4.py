# Ex 4.7.4: reproduce Lab 4.7 convergence tests and apply a quantitative convergence criterion
from common_pt import *
def first_converged(xs, Es, tol):
    """smallest x such that |E(x_k)-E(x_{k+1})| < tol for this and ALL subsequent consecutive pairs"""
    d = np.abs(np.diff(Es))
    for k in range(len(d)):
        if np.all(d[k:] < tol): return xs[k]
    return None
lay = [3,4,5,6,7,8]; El = [E_ads_site('fcc', size=(2,2,n)) for n in lay]
vac = [5,7,9,12];    Ev = [E_ads_site('fcc', vacuum=float(v)) for v in vac]
print("layers:", ", ".join(f"{n}:{e:+.4f}" for n,e in zip(lay,El)))
print("  |consecutive diff| (meV):", np.round(1e3*np.abs(np.diff(El)),2))
print("vacuum:", ", ".join(f"{v}:{e:+.4f}" for v,e in zip(vac,Ev)))
print("  |consecutive diff| (meV):", np.round(1e3*np.abs(np.diff(Ev)),3))
for tol in [10e-3, 1e-3]:
    print(f"tol = {tol*1e3:.0f} meV -> converged from layers = {first_converged(lay,El,tol)}, vacuum = {first_converged(vac,Ev,tol)} A")
