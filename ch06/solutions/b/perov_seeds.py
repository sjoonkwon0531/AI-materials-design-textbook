import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
d=pd.read_csv("../../../ch06/labs/bartel_TableS1.csv").rename(columns={"rA (Ang)":"rA","rB (Ang)":"rB","rX (Ang)":"rX"})
y=(d.exp_label>0).astype(int).values; hal=d.X.isin(["F","Cl","Br","I"]).values; oxi=~hal
print("halide nA counts",d[hal].nA.value_counts().to_dict(),"perov frac by nA",d[hal].groupby("nA").exp_label.apply(lambda s:(s>0).mean()).round(3).to_dict())
raw=d[["rA","rB","rX","nA","nB","nX"]].values
t=(d.rA+d.rX)/(np.sqrt(2)*(d.rB+d.rX)); mu=d.rB/d.rX; nA=d.nA
F=np.c_[t,mu,nA]
for nm,FF in (("raw6",raw),("t,mu,nA",F)):
    print(nm,"hal->ox seeds",[round(RandomForestClassifier(500,min_samples_leaf=2,random_state=s,n_jobs=2).fit(FF[hal],y[hal]).score(FF[oxi],y[oxi]),3) for s in range(5)])
# (t,mu,nA) hal->ox: what does it predict per nA
m=RandomForestClassifier(500,min_samples_leaf=2,random_state=0,n_jobs=2).fit(F[hal],y[hal]); p=m.predict(F[oxi])
for k in sorted(set(nA[oxi])):
    s=(nA[oxi]==k).values; print("ox nA",k,"n",s.sum(),"pred perov frac",p[s].mean().round(3),"true",y[oxi][s].mean().round(3),"acc",(p[s]==y[oxi][s]).mean().round(3))
