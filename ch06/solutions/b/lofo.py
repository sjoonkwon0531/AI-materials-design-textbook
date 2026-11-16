"""Ex 6.7.4: leave-one-family-out CV with the Lab 6.16 LightGBM settings."""
import numpy as np, pandas as pd, lightgbm as lgb, warnings, time
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from pymatgen.core import Composition
warnings.filterwarnings("ignore"); t0=time.time()
df=pd.read_parquet("../../../ch06/labs/67_features_clean.parquet")
feat=[c for c in df.columns if c not in ("formula","target","chemsys","reduced")]
X,y=df[feat].values,df.target.values
P=dict(n_estimators=400,learning_rate=0.05,num_leaves=31,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,n_jobs=2,verbose=-1)
FAM={**{e:"halide" for e in ("F","Cl","Br","I")},"O":"oxide",**{e:"chalcogenide" for e in ("S","Se","Te")},**{e:"pnictide" for e in ("N","P","As","Sb")}}
fam=np.array([FAM.get(max(Composition(f).elements,key=lambda e:e.X if e.X==e.X else 0).symbol,"other") for f in df.formula])
oof=np.zeros_like(y)
# random 5-fold reference (seed 0) for per-family comparison
ref=np.zeros_like(y)
for tr,te in KFold(5,shuffle=True,random_state=0).split(X):
    ref[te]=np.clip(lgb.LGBMRegressor(random_state=0,**P).fit(X[tr],y[tr]).predict(X[te]),0,None)
print(f"{'family':13s} n     mean gap  metal%  train-mean  MAE_LOFO  MSE(signed) nonmetal-MSE  pred mean  MAE(random5)  MAE(predict train mean)")
for f in ("oxide","chalcogenide","halide","pnictide","other"):
    te=fam==f; tr=~te
    p=np.clip(lgb.LGBMRegressor(random_state=0,**P).fit(X[tr],y[tr]).predict(X[te]),0,None); oof[te]=p
    nm=y[te]>0
    print(f"{f:13s} {te.sum():4d}  {y[te].mean():6.2f}   {100*(y[te]==0).mean():5.1f}   {y[tr].mean():6.2f}     {mean_absolute_error(y[te],p):.3f}    {np.mean(p-y[te]):+.3f}      {np.mean(p[nm]-y[te][nm]):+.3f}      {p.mean():.2f}      {mean_absolute_error(y[te],ref[te]):.3f}        {mean_absolute_error(y[te],np.full(te.sum(),y[tr].mean())):.3f}")
    if f=="halide":
        m0=y[te]==0
        print(f"   halide metals: n={m0.sum()} mean pred {p[m0].mean():.2f}; non-metals: n={nm.sum()} mean true {y[te][nm].mean():.2f} mean pred {p[nm].mean():.2f}; frac under-predicted {np.mean(p[nm]<y[te][nm]):.2f}")
        for lo,hi in ((0,1.5),(1.5,3),(3,99)):
            s=nm&(y[te]>=lo)&(y[te]<hi)
            if s.any(): print(f"     true gap in [{lo},{hi}): n={s.sum()} mean signed err {np.mean(p[s]-y[te][s]):+.2f}")
print(f"overall LOFO MAE {mean_absolute_error(y,oof):.3f}; fold mean {np.mean([mean_absolute_error(y[fam==f],oof[fam==f]) for f in set(fam)]):.3f}")
print(f"time {time.time()-t0:.0f}s")
