"""Lab 66_mclca: Monte Carlo propagation of inventory uncertainty in a SYNTHETIC cradle-to-gate LCA.
Functional unit: 1 kg of a hypothetical oxide powder 'X', two synthesis routes.
ALL inventory amounts and emission factors below are SYNTHETIC (illustrative, not from any database).
Only the characterization factors are real: IPCC AR6 WG1 GWP100 (Table 7.15):
CO2 = 1, fossil CH4 = 29.8, N2O = 273 (AR5 values CH4 = 28, N2O = 265 for comparison).
Each uncertain quantity is lognormal with median = deterministic value and geometric SD (GSD)."""
import json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GWP = dict(AR6=dict(CO2=1.0, CH4=29.8, N2O=273.0), AR5=dict(CO2=1.0, CH4=28.0, N2O=265.0))

# ---- background emission factors per unit of input (SYNTHETIC); (median, GSD) ----
# elementary flows in kg per unit; precursors are aggregated kg CO2e per kg
EF = {
    "elec_CO2": (0.45, 1.25), "elec_CH4": (1.0e-3, 2.0), "elec_N2O": (1.0e-5, 2.0),   # per kWh, fossil-heavy grid
    "gas_CO2":  (0.056, 1.05), "gas_CH4": (3.0e-4, 2.0),                                # per MJ heat (incl. upstream leak)
    "P1": (3.0, 1.5), "P2": (9.0, 1.6), "solv": (1.8, 1.4),                            # kg CO2e per kg
}
# ---- foreground inventory per functional unit (SYNTHETIC); (median, GSD) ----
INV = {
    "A": {"P1": (1.10, 1.05), "P2": (0.45, 1.05), "elec": (30.0, 1.3), "gas": (40.0, 1.2),
          "proc_CO2": (0.30, 1.5), "solv": (0.0, 1.0)},        # solid-state route: long high-T calcination
    "B": {"P1": (1.05, 1.05), "P2": (0.45, 1.05), "elec": (12.0, 1.3), "gas": (70.0, 1.2),
          "proc_CO2": (0.05, 1.5), "solv": (0.60, 1.5)},       # sol-gel route: solvent loss, drying heat
}

def ln_sample(rng, med, gsd, n):
    return med * np.exp(np.log(gsd) * rng.standard_normal(n)) if gsd > 1 else np.full(n, med)

def draw_ef(rng, n):
    return {k: ln_sample(rng, *v, n) for k, v in EF.items()}

def gwp(route, ef, inv, cf):
    """Returns dict of contributions (kg CO2e per FU)."""
    e_elec = inv["elec"] * (ef["elec_CO2"] * cf["CO2"] + ef["elec_CH4"] * cf["CH4"] + ef["elec_N2O"] * cf["N2O"])
    e_gas = inv["gas"] * (ef["gas_CO2"] * cf["CO2"] + ef["gas_CH4"] * cf["CH4"])
    return {"P1": inv["P1"] * ef["P1"], "P2": inv["P2"] * ef["P2"], "electricity": e_elec,
            "heat": e_gas, "process CO2": inv["proc_CO2"] * cf["CO2"], "solvent": inv["solv"] * ef["solv"]}

def draw_inv(rng, route, n):
    return {k: ln_sample(rng, *v, n) for k, v in INV[route].items()}

t0 = time.time()
cf = GWP["AR6"]
# deterministic (medians)
det_ef = {k: v[0] for k, v in EF.items()}
det = {r: gwp(r, det_ef, {k: v[0] for k, v in INV[r].items()}, cf) for r in "AB"}
print("=== Deterministic LCIA (medians), kg CO2e per kg X, GWP100 (AR6) ===")
for r in "AB":
    tot = sum(det[r].values())
    print(f"route {r}: total {tot:.2f} | " + ", ".join(f"{k} {v:.2f} ({100*v/tot:.0f}%)" for k, v in det[r].items() if v > 0))
detA5 = {r: sum(gwp(r, det_ef, {k: v[0] for k, v in INV[r].items()}, GWP['AR5']).values()) for r in "AB"}
print(f"with AR5 factors: A {detA5['A']:.2f}, B {detA5['B']:.2f}")

# Monte Carlo, shared background factors (correlated comparison)
N = 100_000
rng = np.random.default_rng(42)
ef = draw_ef(rng, N)
inv = {r: draw_inv(rng, r, N) for r in "AB"}
con = {r: gwp(r, ef, inv[r], cf) for r in "AB"}
tot = {r: sum(con[r].values()) for r in "AB"}
print(f"=== Monte Carlo, N={N}, seed 42 ===")
for r in "AB":
    t = tot[r]; q = np.percentile(t, [2.5, 50, 97.5])
    print(f"route {r}: mean {t.mean():.2f}  median {q[1]:.2f}  95% interval [{q[0]:.2f}, {q[2]:.2f}]  "
          f"sd {t.std():.2f}  (mean/deterministic = {t.mean()/sum(det[r].values()):.3f})")
D = tot["A"] - tot["B"]
print(f"A-B, shared background (correlated): mean {D.mean():.2f}  sd {D.std():.2f}  P(A<B) = {np.mean(D<0):.3f}")
ef2 = draw_ef(rng, N)                           # independent background for B (wrong for a comparison)
totB_ind = sum(gwp("B", ef2, inv["B"], cf).values())
D2 = tot["A"] - totB_ind
print(f"A-B, independent background (wrong):  mean {D2.mean():.2f}  sd {D2.std():.2f}  P(A<B) = {np.mean(D2<0):.3f}")

# sensitivity: squared Spearman rank correlation, normalised (contribution to variance)
from scipy.stats import spearmanr
print("=== Sensitivity of route A total: normalised squared rank correlation ===")
inputs = {**{f"EF {k}": ef[k] for k in EF if k not in ("solv",)}, **{f"inv {k}": inv["A"][k] for k in INV["A"] if k != "solv"}}
rs = {k: spearmanr(v, tot["A"]).statistic for k, v in inputs.items()}
s2 = {k: v**2 for k, v in rs.items()}; S = sum(s2.values())
sens = sorted(((k, rs[k], s2[k] / S) for k in rs), key=lambda z: -z[2])
for k, r_, c in sens[:6]:
    print(f"  {k:12s} rank corr {r_:+.3f}  share {100*c:5.1f}%")

# Derivation-box check: single product term ln-variances add
e_el = inv["A"]["elec"] * ef["elec_CO2"]
gsd_mc = np.exp(np.log(e_el).std())
gsd_th = np.exp(np.sqrt(np.log(1.3)**2 + np.log(1.25)**2))
print(f"electricity CO2 term of A: GSD (MC) = {gsd_mc:.4f}, sqrt(ln^2 1.3 + ln^2 1.25) rule = {gsd_th:.4f}")
med_th = 30 * 0.45; mean_th = med_th * np.exp(0.5 * np.log(gsd_th)**2)
print(f"  median {np.median(e_el):.3f} (theory {med_th:.3f}),  mean {e_el.mean():.3f} (theory {mean_th:.3f})")

# grid scenario: low-carbon electricity (SYNTHETIC: 0.05 kg CO2/kWh, CH4 and N2O scaled by 1/9)
rng = np.random.default_rng(7)
EF_low = dict(EF); EF_low["elec_CO2"] = (0.05, 1.25); EF_low["elec_CH4"] = (1.0e-3 / 9, 2.0); EF_low["elec_N2O"] = (1.0e-5 / 9, 2.0)
efl = {k: ln_sample(rng, *v, N) for k, v in EF_low.items()}
totl = {r: sum(gwp(r, efl, draw_inv(rng, r, N), cf).values()) for r in "AB"}
Dl = totl["A"] - totl["B"]
print(f"=== low-carbon grid scenario: A median {np.median(totl['A']):.2f}, B median {np.median(totl['B']):.2f}, "
      f"P(A<B) = {np.mean(Dl<0):.3f}")

# convergence of the MC mean (route A): standard error vs N
print("=== MC convergence (route A mean) ===")
for n in (100, 1000, 10000, 100000):
    means = [sum(gwp("A", draw_ef(np.random.default_rng(1000 + s), n), draw_inv(np.random.default_rng(2000 + s), "A", n), cf).values()).mean()
             for s in range(20)]
    print(f"  N={n:6d}: sd of mean over 20 seeds = {np.std(means, ddof=1):.4f}   sd/sqrt(N) estimate = {tot['A'].std()/np.sqrt(n):.4f}")
print(f"time {time.time()-t0:.1f} s")
json.dump(dict(totA=tot["A"][:20000].tolist(), totB=tot["B"][:20000].tolist(), D=D[:20000].tolist(),
               D2=D2[:20000].tolist(), sens=[(k, float(r_), float(c)) for k, r_, c in sens],
               det={r: {k: float(v) for k, v in det[r].items()} for r in "AB"},
               totlA=totl["A"][:20000].tolist(), totlB=totl["B"][:20000].tolist()),
          open(os.path.join(HERE, "66_mclca.json"), "w"))
