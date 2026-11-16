"""Ex 6.6.4: modifications of the synthetic MC-LCA (same SYNTHETIC inventory as Lab 6.14)."""
import numpy as np, copy
from scipy.stats import spearmanr
GWP=dict(CO2=1.0,CH4=29.8,N2O=273.0)
EF0={"elec_CO2":(0.45,1.25),"elec_CH4":(1.0e-3,2.0),"elec_N2O":(1.0e-5,2.0),"gas_CO2":(0.056,1.05),"gas_CH4":(3.0e-4,2.0),
     "P1":(3.0,1.5),"P2":(9.0,1.6),"solv":(1.8,1.4)}
INV0={"A":{"P1":(1.10,1.05),"P2":(0.45,1.05),"elec":(30.0,1.3),"gas":(40.0,1.2),"proc_CO2":(0.30,1.5),"solv":(0.0,1.0)},
      "B":{"P1":(1.05,1.05),"P2":(0.45,1.05),"elec":(12.0,1.3),"gas":(70.0,1.2),"proc_CO2":(0.05,1.5),"solv":(0.60,1.5)}}
def ln_sample(rng,med,gsd,n): return med*np.exp(np.log(gsd)*rng.standard_normal(n)) if gsd>1 else np.full(n,med)
def gwp(ef,inv,cf=GWP):
    e_el=inv["elec"]*(ef["elec_CO2"]*cf["CO2"]+ef["elec_CH4"]*cf["CH4"]+ef["elec_N2O"]*cf["N2O"])
    e_g=inv["gas"]*(ef["gas_CO2"]*cf["CO2"]+ef["gas_CH4"]*cf["CH4"])
    return inv["P1"]*ef["P1"]+inv["P2"]*ef["P2"]+e_el+e_g+inv["proc_CO2"]*cf["CO2"]+inv["solv"]*ef["solv"]
def run(EF,INV,shared_meter=False,label="",N=100_000,seed=42):
    rng=np.random.default_rng(seed)
    ef={k:ln_sample(rng,*v,N) for k,v in EF.items()}
    inv={r:{k:ln_sample(rng,*v,N) for k,v in INV[r].items()} for r in "AB"}
    if shared_meter:
        z=np.random.default_rng(seed+1).standard_normal(N)
        for r in "AB": inv[r]["elec"]=INV[r]["elec"][0]*np.exp(np.log(INV[r]["elec"][1])*z)
    tA,tB=gwp(ef,inv["A"]),gwp(ef,inv["B"]); D=tA-tB
    qA=np.percentile(tA,[2.5,50,97.5])
    inputs={**{f"EF {k}":ef[k] for k in EF if k!="solv"},**{f"inv {k}":inv["A"][k] for k in INV["A"] if k!="solv"}}
    rs={k:spearmanr(v,tA).statistic for k,v in inputs.items()}; S=sum(v**2 for v in rs.values())
    sens=sorted(((k,rs[k],rs[k]**2/S) for k in rs),key=lambda z:-z[2])[:5]
    print(f"--- {label}")
    print(f"  A: mean {tA.mean():.2f} median {qA[1]:.2f} 95% [{qA[0]:.2f}, {qA[2]:.2f}] width {qA[2]-qA[0]:.2f} sd {tA.std():.2f}")
    print(f"  A-B: mean {D.mean():.2f} sd {D.std():.2f} P(A<B) {np.mean(D<0):.3f}; corr(A,B) {np.corrcoef(tA,tB)[0,1]:.3f}")
    print("  sens:", "; ".join(f"{k} {100*c:.1f}%" for k,r,c in sens))
    return tA,tB
run(EF0,INV0,label="baseline (Lab 6.14)")
I=copy.deepcopy(INV0); I["A"]["elec"]=(30.0,1.1); I["B"]["elec"]=(12.0,1.1)
run(EF0,I,label="(a) elec GSD 1.3->1.1 (both routes)")
I=copy.deepcopy(INV0); I["A"]["elec"]=(30.0,1.1)
run(EF0,I,label="(a') elec GSD 1.1 for route A only")
E=dict(EF0); E["P2"]=(9.0,2.0)
run(E,INV0,label="(b) EF P2 GSD 1.6->2.0")
run(EF0,INV0,shared_meter=True,label="(c) shared meter error z for elec of A and B")
# analytic check for (c): Var contribution of elec usage terms
# variance of A-B sd sources
for sm in (False,True):
    pass
