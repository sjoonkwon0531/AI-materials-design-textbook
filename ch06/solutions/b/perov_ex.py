"""Ex 6.8.3 and 6.8.4 on Bartel Table S1."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss
warnings.filterwarnings("ignore")
d=pd.read_csv("../ch06/labs/bartel_TableS1.csv").rename(columns={"rA (Ang)":"rA","rB (Ang)":"rB","rX (Ang)":"rX"})
y=(d.exp_label>0).astype(int).values; tr=(d.is_train==1).values; te=(d.is_train==-1).values
rA,rB,rX,nA,nB,nX=[d[c].values.astype(float) for c in ("rA","rB","rX","nA","nB","nX")]
t=(rA+rX)/(np.sqrt(2)*(rB+rX)); mu=rB/rX; tau=rX/rB-nA*(nA-(rA/rB)/np.log(rA/rB))
print("== 6.8.3")
eps=1e-3
m1=LogisticRegression(C=1e6).fit(tau[tr,None],y[tr]); P1=m1.predict_proba(tau[:,None])[:,1]
m2=LogisticRegression(C=1e6).fit(np.log(tau[tr,None]),y[tr]); P2=m2.predict_proba(np.log(tau[:,None]))[:,1]
iso=IsotonicRegression(out_of_bounds="clip",increasing=False).fit(tau[tr],y[tr]); P3=iso.predict(tau)
print(f"logistic tau: a={m1.coef_[0,0]:.3f} b={m1.intercept_[0]:.3f}; P=0.5 at tau={-m1.intercept_[0]/m1.coef_[0,0]:.3f}")
print(f"logistic ln tau: a={m2.coef_[0,0]:.3f} b={m2.intercept_[0]:.3f}; P=0.5 at tau={np.exp(-m2.intercept_[0]/m2.coef_[0,0]):.3f}")
print("isotonic: number of distinct levels", len(np.unique(P3[tr])), "levels", np.round(np.unique(iso.y_thresholds_),3))
# where isotonic jumps
xt,yt=iso.X_thresholds_,iso.y_thresholds_
print("isotonic steps (tau, P):", [(round(a,2),round(b,3)) for a,b in zip(xt,yt)][:40])
res={}
for name,P in (("logistic(tau)",P1),("logistic(ln tau)",P2),("isotonic(tau)",P3)):
    Pc=np.clip(P,eps,1-eps)
    res[name]=(brier_score_loss(y[te],P[te]),log_loss(y[te],Pc[te]),brier_score_loss(y[tr],P[tr]),log_loss(y[tr],Pc[tr]))
    print(f"{name:17s} test Brier {res[name][0]:.4f} logloss {res[name][1]:.4f} | train Brier {res[name][2]:.4f} logloss {res[name][3]:.4f}; test acc(P>0.5) {np.mean((P[te]>0.5)==y[te]):.3f}")
    print("   isotonic test P exactly 0 or 1 (before clip):", int(np.sum((P[te]==0)|(P[te]==1)))) if "iso" in name else None
    for lo,hi in ((0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)):
        m=(P>=lo)&(P<hi)
        print(f"   P in [{lo:.1f},{hi:.1f}): n={m.sum():3d} mean P {P[m].mean() if m.any() else float('nan'):.2f} observed {y[m].mean() if m.any() else float('nan'):.2f}")
# steepness near threshold
for tv in (4.09,4.29):
    print(f"tau={tv}: P logistic {m1.predict_proba([[tv]])[0,1]:.3f}, ln-logistic {m2.predict_proba([[np.log(tv)]])[0,1]:.3f}, isotonic {iso.predict([tv])[0]:.3f}")
# bootstrap the test-set difference in log-loss between ln-logistic and logistic
rng=np.random.default_rng(0); it=np.where(te)[0]; dl=[];db=[]
for b in range(2000):
    s=rng.choice(it,len(it),replace=True)
    l=lambda P:log_loss(y[s],np.clip(P[s],eps,1-eps),labels=[0,1])
    dl.append(l(P2)-l(P1)); db.append(brier_score_loss(y[s],P2[s])-brier_score_loss(y[s],P1[s]))
print(f"bootstrap test (ln-logistic minus logistic): dlogloss {np.mean(dl):.4f} [{np.percentile(dl,2.5):.4f},{np.percentile(dl,97.5):.4f}], dBrier {np.mean(db):.4f} [{np.percentile(db,2.5):.4f},{np.percentile(db,97.5):.4f}]")
dl=[]
for b in range(2000):
    s=rng.choice(it,len(it),replace=True)
    dl.append(log_loss(y[s],np.clip(P3[s],eps,1-eps),labels=[0,1])-log_loss(y[s],P1[s],labels=[0,1]))
print(f"bootstrap test (isotonic minus logistic): dlogloss {np.mean(dl):.4f} [{np.percentile(dl,2.5):.4f},{np.percentile(dl,97.5):.4f}]")
print("== 6.8.4")
hal=d.X.isin(["F","Cl","Br","I"]).values; oxi=~hal
sets={"raw 6":np.c_[rA,rB,rX,nA,nB,nX],"ratios (rA/rX, rB/rX, nA, nB)":np.c_[rA/rX,rB/rX,nA,nB],
      "(t, mu)":np.c_[t,mu],"(t, mu, nA)":np.c_[t,mu,nA],"(t, mu, nA, nB)":np.c_[t,mu,nA,nB],"(tau)":tau[:,None]}
for name,F in sets.items():
    out=[]
    for a_,b_ in ((oxi,hal),(hal,oxi)):
        accs=[RandomForestClassifier(500,min_samples_leaf=2,random_state=s,n_jobs=2).fit(F[a_],y[a_]).score(F[b_],y[b_]) for s in range(5)]
        out.append((accs[0],np.mean(accs),np.std(accs)))
    print(f"{name:32s} ox->hal {out[0][0]:.3f} (5 seeds {out[0][1]:.3f}+/-{out[0][2]:.3f})   hal->ox {out[1][0]:.3f} (5 seeds {out[1][1]:.3f}+/-{out[1][2]:.3f})")
print("ranges: oxides t %.2f-%.2f mu %.2f-%.2f nA %s; halides t %.2f-%.2f mu %.2f-%.2f nA %s"%(t[oxi].min(),t[oxi].max(),mu[oxi].min(),mu[oxi].max(),sorted(set(nA[oxi])),t[hal].min(),t[hal].max(),mu[hal].min(),mu[hal].max(),sorted(set(nA[hal]))))
print("perovskite fraction: oxides %.3f halides %.3f; by nA in oxides:"%(y[oxi].mean(),y[hal].mean()), {int(k):round(y[oxi&(nA==k)].mean(),3) for k in sorted(set(nA[oxi]))}, "halides nA=1:",round(y[hal].mean(),3))
print("oxide counts by nA", {int(k):int((oxi&(nA==k)).sum()) for k in sorted(set(nA[oxi]))})
# t-window rule per family for reference
pt=((t>0.825)&(t<1.059)); print("t window acc: oxides %.3f halides %.3f"%((pt==y)[oxi].mean(),(pt==y)[hal].mean()))
