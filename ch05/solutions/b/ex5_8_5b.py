"""Ex 5.8.5 (cont.): lateral-size dependence of the apparent interfacial width of field-free bulk lamellae
(capillary-wave test). Seeds given on the command line; default 3 4. Prints erf width w and tanh length per run,
then fits w^2 = c + s ln(L) over all runs found in the output of this and ex5_8_5.py (seed 2)."""
import sys, time, math, numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
sys.path.insert(0, "../../../ch05/labs")
from dpd_core import seed, forces, run, run_field, make_amat
from bcp_common import build_melt
N = 10; D = 6.41; NB = 80; A = make_amat(25.0, 15.0)
def width(LY, sd):
    Lx = 2*D; box = np.array([Lx, LY, LY]); nch = int(round(3*Lx*LY*LY/N))
    rng = np.random.default_rng(sd); seed(sd)
    x, v, typ, bonds = build_melt(nch, N, 0.5, box, rng); e = np.where(typ == 0, 1.0, -1.0)
    f = np.zeros_like(x); pt = np.zeros(3)
    forces(x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0, f, pt)
    run_field(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1500, 10**6, pt, 1.0, 2*np.pi*2/Lx)
    run(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1500, 10**6, pt)
    edges = np.linspace(-D/2, D/2, NB+1); xc = 0.5*(edges[1:]+edges[:-1]); hA = np.zeros(NB); hB = np.zeros(NB)
    for blk in range(160):
        run(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 50, 10**6, pt)
        ph = np.angle(np.sum(e*np.exp(1j*2*np.pi*x[:, 0]/D)))
        xs = ((x[:, 0] - ph*D/(2*np.pi) + D/2) % D) - D/2
        hA += np.histogram(xs[typ == 0], edges)[0]; hB += np.histogram(xs[typ == 1], edges)[0]
    psi = (hA-hB)/(hA+hB)
    pe, _ = curve_fit(lambda X, a, w: a*(erf((X+D/4)/(np.sqrt(2)*w)) - erf((X-D/4)/(np.sqrt(2)*w)) - 1), xc, psi, p0=(1, 0.3))
    pt_, _ = curve_fit(lambda X, a, l: a*(np.tanh((X+D/4)/l) - np.tanh((X-D/4)/l) - 1), xc, psi, p0=(1, 0.3))
    return pe[1], pt_[1]
if __name__ == "__main__":
    t0 = time.time(); seeds = [int(s) for s in sys.argv[1:]] or [3, 4]
    res = {4.0: [0.3157], 6.0: [0.3712], 12.0: [0.4108]}         # seed 2 values from ex5_8_5.py
    for sd in seeds:
        for LY in (4.0, 6.0, 12.0):
            w, lam = width(LY, sd); res[LY].append(w)
            print(f"seed {sd}, lateral {LY:.0f}x{LY:.0f}: erf w = {w:.4f}, tanh lambda = {lam:.4f}  ({time.time()-t0:.0f} s)", flush=True)
    Ls = np.array(sorted(res)); m = np.array([np.mean(np.square(res[L])) for L in Ls])
    se = np.array([np.std(np.square(res[L]), ddof=1)/np.sqrt(len(res[L])) for L in Ls])
    for L, mm, s in zip(Ls, m, se):
        print(f"L={L:.0f}: <w^2> = {mm:.4f} +/- {s:.4f}  (w_rms = {np.sqrt(mm):.4f}, n={len(res[L])})")
    p, cov = np.polyfit(np.log(Ls), m, 1, w=1/se, cov="unscaled")
    s, ds = p[0], np.sqrt(cov[0, 0])
    print(f"fit <w^2> = {p[1]:.4f} + {s:.4f} ln L  (slope error {ds:.4f}) -> capillary tension kT/(2 pi slope) = {1/(2*np.pi*s):.2f}"
          f" [{1/(2*np.pi*(s+ds)):.2f}, {1/(2*np.pi*max(s-ds,1e-9)):.2f}];  Helfand-Tagami rho a sqrt(chi/6) = {3*0.990*math.sqrt(0.286*15/6):.2f}")
    for xi in (0.5, 1.0):
        w0 = [np.sqrt(max(mm - 1/(2*np.pi*3*0.990*math.sqrt(0.286*15/6))*np.log(L/xi), 0)) for L, mm in zip(Ls, m)]
        print(f"intrinsic w0 = sqrt(<w^2> - (kT/2 pi gamma_HT) ln(L/xi)) with xi={xi}: " + ", ".join(f"{v:.3f}" for v in w0))
    print(f"wall {time.time()-t0:.0f} s")
