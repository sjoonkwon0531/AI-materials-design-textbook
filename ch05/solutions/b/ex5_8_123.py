"""Ex 5.8.1 (Nbar, ODT), 5.8.2 (SST period), 5.8.3 (square-wave harmonics), 5.8.6(a) (stretch/compress cost)."""
import numpy as np
print("5.8.1")
for N, Re2 in ((10, 8.81), (10, 8.84), (20, 18.63)):
    Nb = (3*Re2**1.5/N)**2
    fh = 10.495 + 41.0*Nb**(-1/3); gl = fh + 123.0*Nb**(-0.56)
    print(f"  N={N} <Re^2>={Re2}: Nbar={Nb:.1f}, FH (chiN)_ODT={fh:.2f} -> chi={fh/N:.3f}, Delta a={fh/N/0.286:.2f}; "
          f"Glaser (chiN)_ODT={gl:.2f} -> chi={gl/N:.3f}, Delta a={gl/N/0.286:.2f}")
print("5.8.2")
pref = (32/np.pi**2)**(1/3)*6**(-1/6); print(f"  prefactor (32/pi^2)^(1/3) 6^(-1/6) = {pref:.4f}")
a = 0.70
for chi, N in ((0.04, 500), (0.16, 125)):
    L0 = pref*a*chi**(1/6)*N**(2/3)
    print(f"  chi={chi}, N={N}: chiN={chi*N:.0f}, L0={L0:.2f} nm, Re0=a sqrt(N)={a*np.sqrt(N):.2f} nm, L0/Re0={L0/(a*np.sqrt(N)):.3f}, "
          f"HT interface width a/sqrt(6chi)={a/np.sqrt(6*chi):.3f} nm (ratio to L0 {a/np.sqrt(6*chi)/L0:.3f})")
print(f"  ratio L0(0.16)/L0(0.04) = {(0.16/0.04)**(1/6)*(125/500)**(2/3):.4f} ; (0.16/0.04)^(-1/2) = {(0.16/0.04)**-0.5:.4f}")
print("5.8.3")
def psi_m(f, m, D=1.0, n=200000):
    x = (np.arange(n)+0.5)/n*D; psi = np.where(x < f*D, 1.0, -1.0)
    return np.mean(psi*np.exp(-2j*np.pi*m*x/D))
for f in (0.5, 1/3):
    num = [abs(psi_m(f, m))**2 for m in range(1, 7)]
    ana = [(2*np.sin(np.pi*m*f)/(np.pi*m))**2 for m in range(1, 7)]
    print(f"  f={f:.4f}: |psi_m|^2 numeric " + ", ".join(f"{v:.5f}" for v in num))
    print(f"           analytic (2 sin(pi m f)/(pi m))^2 " + ", ".join(f"{v:.5f}" for v in ana))
    print(f"           S(2q0)/S(q0)={ana[1]/ana[0]:.4f}, S(3q0)/S(q0)={ana[2]/ana[0]:.2e}, psi_0={2*f-1:+.4f}")
print("5.8.6(a)")
g = lambda x: (x**2 + 2/x)/3
for x in (0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 1.7):
    print(f"  x={x}: F/Fmin = {g(x):.5f} (+{(g(x)-1)*100:.2f}%)")
N, a, D0, chi = 10, 0.990, 6.41, 0.286*15
Fel = np.pi**2*D0**2/(32*N*a**2); Fint = 2*N*a*np.sqrt(chi/6)/D0
print(f"  N=10 DPD (a={a}, chi={chi:.2f}): SST at D=6.41: F_el={Fel:.3f} kT, F_int={Fint:.3f} kT per chain; "
      f"SST optimum D0={(32*a**3*N**2*np.sqrt(chi/6)/np.pi**2)**(1/3):.3f}, F_min=3 F_el(D0)="
      f"{3*np.pi**2*((32*a**3*N**2*np.sqrt(chi/6)/np.pi**2)**(1/3))**2/(32*N*a**2):.3f} kT")
