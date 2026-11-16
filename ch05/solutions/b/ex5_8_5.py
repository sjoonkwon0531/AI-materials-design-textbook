"""Ex 5.8.5: interfacial width of DPD A5B5 lamellae (Delta a = 15).
(A) Lab 5.15 dense-pattern film (same seed/protocol to t=400), then t=400..800 sampled every t=5:
    time-averaged S(m q0)/n (m=1..4) and the composition profile psi(x) folded on the pattern period.
(B) Bulk lamellae without any field (Lab 5.14 box 2 L0 x 6 x 6, N=10, D = 6.41): profile phase-aligned
    by the first-harmonic phase of each snapshot.
Fits: tanh profile psi = A[tanh((x+L0/4)/lam) - tanh((x-L0/4)/lam) - 1] and the Gaussian-smeared square wave
(erf profile) with std w; harmonic-ratio estimate of w."""
import sys, time, math, numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
sys.path.insert(0, "../ch05/labs")
from dpd_core import seed, forces, run, run_field, make_amat
from bcp_common import build_melt, sq_psi
L0 = 6.41; N = 10; DA = 15.0; NB = 80
t0 = time.time()
def model_tanh(x, A, lam): return A*(np.tanh((x+L0/4)/lam) - np.tanh((x-L0/4)/lam) - 1)
def model_erf(x, A, w): return A*(erf((x+L0/4)/(np.sqrt(2)*w)) - erf((x-L0/4)/(np.sqrt(2)*w)) - 1)
def analyse(tag, xs_list, typ, S):
    edges = np.linspace(-L0/2, L0/2, NB+1); xc = 0.5*(edges[1:]+edges[:-1])
    hA = np.zeros(NB); hB = np.zeros(NB)
    for xs in xs_list:
        hA += np.histogram(xs[typ == 0], edges)[0]; hB += np.histogram(xs[typ == 1], edges)[0]
    psi = (hA-hB)/(hA+hB); rho = (hA+hB)/(hA+hB).mean()
    psin = (hA-hB)/(hA+hB).mean()                        # number-weighted (what S(q) actually sees)
    cm = [abs(np.mean(psin*np.exp(-2j*np.pi*m*xc/L0)))**2 for m in (1, 3)]
    cc = [abs(np.mean(psi*np.exp(-2j*np.pi*m*xc/L0)))**2 for m in (1, 3)]
    print(f"[{tag}] |Fourier|^2 of composition psi: m=1 {cc[0]:.4f}, m=3 {cc[1]:.4f};  of number-weighted rho*psi/<rho>: "
          f"m=1 {cm[0]:.4f}, m=3 {cm[1]:.4f};  density in domain centres/mean = {rho[NB//2]:.3f}, {rho[0]:.3f}")
    pt, _ = curve_fit(model_tanh, xc, psi, p0=(1, 0.3)); pe, _ = curve_fit(model_erf, xc, psi, p0=(1, 0.3))
    rt = np.sqrt(np.mean((model_tanh(xc, *pt)-psi)**2)); re = np.sqrt(np.mean((model_erf(xc, *pe)-psi)**2))
    print(f"[{tag}] psi(centre of A)={psi[NB//2]:.4f}, psi(centre of B)={psi[0]:.4f}; total density at interface/mean = {rho[NB//4]:.3f}, {rho[3*NB//4]:.3f}")
    print(f"[{tag}] tanh fit: A={pt[0]:.4f}, lambda={pt[1]:.4f} (rms {rt:.4f});  erf fit: A={pe[0]:.4f}, w={pe[1]:.4f} (rms {re:.4f})")
    if S is not None:
        q0 = 2*np.pi/L0; Sm = S
        print(f"[{tag}] time-averaged S(m q0)/n, m=1..4: " + ", ".join(f"{s:.4f}" for s in Sm))
        w13 = math.sqrt(math.log((1/9)/(Sm[2]/Sm[0]))/(8*q0*q0))
        w1 = math.sqrt(-math.log(Sm[0]/(4/np.pi**2)))/q0
        print(f"[{tag}] w from S3/S1 = {w13:.4f};  w from S1 alone (amplitude 1 assumed) = {w1:.4f}")
        # harmonics predicted by the fitted profiles (sharp-wave amplitude 2/pi * A)
        for nm, A_, fac in (("tanh", pt[0], lambda k: (np.pi*k*pt[1]/2)/np.sinh(np.pi*k*pt[1]/2)),
                            ("erf", pe[0], lambda k: np.exp(-k*k*pe[1]**2/2))):
            pred = [(A_*2*np.sin(np.pi*m/2)/(np.pi*m)*fac(m*q0))**2 for m in (1, 3)]
            print(f"[{tag}]   profile-fit ({nm}) predicts S(q0)/n={pred[0]:.4f}, S(3q0)/n={pred[1]:.4f}")
    return xc, psi
# ---------------- (A) film with dense pattern
box = np.array([4*L0, 4*L0, 3.0]); nch = int(round(3*box.prod()/N))
rng = np.random.default_rng(5); seed(5)
x, v, typ, bonds = build_melt(nch, N, 0.5, box, rng); e = np.where(typ == 0, 1.0, -1.0)
A = make_amat(25.0, DA); f = np.zeros_like(x); pt3 = np.zeros(3)
forces(x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0, f, pt3)
k = 2*np.pi/L0
for blk in range(8):
    run_field(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1250, 10**6, pt3, 0.2, k, 0, 1.0)
qq = np.array([[m*k, 0, 0] for m in (1, 2, 3, 4)])
s400 = sq_psi(x, e, qq)/len(x)
print(f"[film] t=400 snapshot S(m q0)/n: " + ", ".join(f"{s:.4f}" for s in s400) + f"  ({time.time()-t0:.0f} s)")
S = np.zeros(4); xs_list = []
for blk in range(80):
    run_field(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 125, 10**6, pt3, 0.2, k, 0, 1.0)
    S += sq_psi(x, e, qq)/len(x); xs_list.append(((x[:, 0] + L0/2) % L0) - L0/2)
S /= 80
analyse("film, dense pattern h=0.2, t=400-800", xs_list, typ, S)
print(f"  ({time.time()-t0:.0f} s)", flush=True)
# ---------------- (B) bulk lamellae, no field, lateral size LY = 4, 6, 12
for LY in (4.0, 6.0, 12.0):
  D = 6.41; Lx = 2*D; boxb = np.array([Lx, LY, LY]); nch = int(round(3*Lx*LY*LY/N))
  rng = np.random.default_rng(2); seed(2)
  x, v, typ, bonds = build_melt(nch, N, 0.5, boxb, rng); e = np.where(typ == 0, 1.0, -1.0)
  f = np.zeros_like(x); pt3 = np.zeros(3)
  forces(x, v, typ, A, 4.5, 3.0, 0.04, boxb, bonds, 4.0, f, pt3)
  run_field(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, boxb, bonds, 4.0, 1500, 10**6, pt3, 1.0, 2*np.pi*2/Lx)
  run(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, boxb, bonds, 4.0, 1500, 10**6, pt3)
  S = np.zeros(4); xs_list = []; qb = np.array([[m*2*np.pi/D, 0, 0] for m in (1, 2, 3, 4)])
  for blk in range(160):
      run(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, boxb, bonds, 4.0, 50, 10**6, pt3)
      c = np.sum(e*np.exp(1j*2*np.pi*x[:, 0]/D)); ph = np.angle(c)          # A-rich centre at x = ph D/2pi
      xs_list.append(((x[:, 0] - ph*D/(2*np.pi) + D/2) % D) - D/2)
      S += sq_psi(x, e, qb)/len(x)
  S /= 160
  analyse(f"bulk lamellae, no field, lateral {LY:.0f}x{LY:.0f}, t=0-320", xs_list, typ, S); print(f"  ({time.time()-t0:.0f} s)", flush=True)
lam_HT = 0.990/math.sqrt(6*0.286*15)
q0 = 2*np.pi/L0; w_ex = math.sqrt(math.log((1/9)/(0.0204/0.3882))/(8*q0*q0))
print(f"exercise numbers (Lab 5.15 single snapshot S1=0.3882, S3=0.0204): w = {w_ex:.4f}")
gam_HT = 3*0.990*math.sqrt(0.286*15/6)
print(f"HT tension gamma = rho a sqrt(chi/6) = {gam_HT:.3f} kT per unit area")
print(f"Helfand-Tagami lambda = a/sqrt(6 chi) = {lam_HT:.4f} (chi=4.29); with chi=15/3.27: {0.990/math.sqrt(6*15/3.27):.4f}")
print(f"equivalent Gaussian std of a tanh(z/lam) profile: slope-matched w = 2 lam/sqrt(2 pi) = {2*lam_HT/math.sqrt(2*math.pi):.4f}, "
      f"small-k Fourier-matched w = pi lam/sqrt(12) = {math.pi*lam_HT/math.sqrt(12):.4f}")
print(f"wall {time.time()-t0:.0f} s")
