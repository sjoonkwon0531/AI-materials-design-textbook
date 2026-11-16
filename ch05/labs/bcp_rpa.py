"""Lab 5.8A - Calibrating the effective chi of DPD A5B5 melts against Leibler's RPA.
Disordered melts, cubic box L=10 (300 chains, 3000 beads), Delta a = 0..4.
S_psi(q) measured on all lattice q-vectors with |n| <= 4, shell-averaged; intrachain block form factors
s_AA, s_BB, s_AB measured from the SAME simulation, so the only free parameter of the RPA is chi_eff.
Run: python bcp_rpa.py   (about 2-3 min)"""
import os, json, math, time
import numpy as np
from scipy.optimize import minimize_scalar
from dpd_core import seed, forces, run, make_amat
from bcp_common import build_melt, qvectors, sq_psi, radial, chain_stats, intrachain_sq, rpa_spsi

OUT = os.path.dirname(os.path.abspath(__file__))
N, NA, L = 10, 5, 10.0

def melt_run(da, sd, t_eq=40.0, t_prod=240.0):
    box = np.array([L, L, L]); nch = int(3 * L ** 3 / N)
    rng = np.random.default_rng(sd); seed(sd)
    x, v, typ, bonds = build_melt(nch, N, 0.5, box, rng); e = np.where(typ == 0, 1.0, -1.0)
    A = make_amat(25.0, da); f = np.zeros_like(x); pt = np.zeros(3)
    forces(x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0, f, pt)
    run(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, int(t_eq / 0.04), 10 ** 6, pt)
    qv = qvectors(box, 4); S = np.zeros(len(qv)); ns = 0
    qfine = np.linspace(0.2, 3.0, 57); ff = np.zeros((len(qfine), 3)); nf = 0; st = np.zeros(3)
    for blk in range(int(t_prod / 4.0)):
        run(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 100, 10 ** 6, pt)
        S += sq_psi(x, e, qv); ns += 1
        if blk % 10 == 9:
            ff += intrachain_sq(x, nch, N, NA, box, qfine, rng=rng); nf += 1
            st += np.array(chain_stats(x, nch, N, box))
    uq, Sr = radial(qv, S / ns)
    return uq, Sr, qfine, ff / nf, st / nf

if __name__ == "__main__":
    t0 = time.time(); out = {}
    print("=== disordered A5B5 melts: RPA fit of chi_eff (only free parameter) ===")
    rows = []
    for da in (0.0, 1.0, 2.0, 3.0, 4.0):
        uq, Sr, qf, ff, st = melt_run(da, sd=21)
        saa, sbb, sab = [np.interp(uq, qf, ff[:, k]) for k in range(3)]
        m = uq <= 2.6
        def loss(c):
            return np.sum((np.log(rpa_spsi(saa[m], sbb[m], sab[m], c)) - np.log(Sr[m])) ** 2)
        chi = minimize_scalar(loss, bounds=(-0.5, 0.95), method="bounded").x
        rms = math.sqrt(loss(chi) / m.sum())
        kq = np.argmax(Sr)
        rows.append(dict(da=da, chi=chi, rms=rms, qpk=float(uq[kq]), Spk=float(Sr[kq]), Re2=st[0], Rg2=st[1], b2=st[2],
                         uq=uq.tolist(), Sr=Sr.tolist(), qf=qf.tolist(), ff=ff.tolist()))
        print(f"  Delta a={da:3.1f}: chi_eff={chi:+.4f}  (rms log-residual {rms:.3f});  sim peak S={Sr[kq]:.2f} at q={uq[kq]:.3f};"
              f"  <Re^2>={st[0]:.2f} <Rg^2>={st[1]:.3f} <b^2>={st[2]:.3f}")
    das = np.array([r["da"] for r in rows]); chis = np.array([r["chi"] for r in rows])
    p = np.polyfit(das, chis, 1)
    print(f"linear fit: chi_eff = {p[0]:.3f} Delta a {p[1]:+.3f}   (monomer mapping 0.286; polymer mapping 1/3.27 = 0.306)")
    # mean-field spinodal from the athermal form factors: Gamma(q) - 2 chi = 0 at its minimum
    r0 = rows[0]; qf = np.array(r0["qf"]); ff = np.array(r0["ff"])
    gam = (ff[:, 0] + ff[:, 1] + 2 * ff[:, 2]) / (ff[:, 0] * ff[:, 1] - ff[:, 2] ** 2)
    k = np.argmin(gam); chis_ = gam[k] / 2; Rg = math.sqrt(r0["Rg2"])
    print(f"RPA (measured form factors, Delta a=0): min Gamma at q*={qf[k]:.2f} -> q* Rg = {qf[k]*Rg:.2f} "
          f"(continuous Gaussian chain: 1.95);  spinodal chi_s N = {chis_*N:.2f} (Leibler: 10.495)")
    das_s = (chis_ - p[1]) / p[0]
    Re = math.sqrt(r0["Re2"]); Nbar = (3.0 * Re ** 3 / N) ** 2
    fh = 10.495 + 41.0 * Nbar ** (-1 / 3); gl = fh + 123.0 * Nbar ** (-0.56)
    print(f"mean-field spinodal at Delta a = {das_s:.2f};  Nbar = (rho Re^3/N)^2 = {Nbar:.1f};"
          f"  FH (chi N)_ODT = {fh:.1f},  Glaser et al. fit = {gl:.1f}")
    out.update(rows=rows, fit=p.tolist(), chis=chis_, qstar=float(qf[k]), Nbar=Nbar, fh=fh, gl=gl, da_spin=das_s)
    json.dump(out, open(os.path.join(OUT, "bcp_rpa.json"), "w"))
    print(f"wall time {time.time()-t0:.0f} s")
