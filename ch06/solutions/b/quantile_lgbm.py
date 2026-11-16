"""Ex 6.5.6(b): LightGBM quantile regression (alpha 0.05, 0.95) -> 90% interval, 5-fold CV seed 0 on expt_gap Magpie."""
import numpy as np, pandas as pd, lightgbm as lgb, time
from sklearn.model_selection import KFold
L="../ch06/labs/"
df=pd.read_parquet(L+"magpie_expt_gap.parquet")
X=df.drop(columns=["formula","target"]).to_numpy(np.float32); y=df["target"].to_numpy()
P=dict(n_estimators=600,learning_rate=0.05,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,
       min_child_samples=10,n_jobs=1,random_state=0,verbose=-1)
t0=time.time(); lo=np.zeros(len(y)); hi=np.zeros(len(y)); md=np.zeros(len(y))
for tr,te in KFold(5,shuffle=True,random_state=0).split(X):
    for a,arr in ((0.05,lo),(0.95,hi),(0.5,md)):
        arr[te]=lgb.LGBMRegressor(objective="quantile",alpha=a,**P).fit(X[tr],y[tr]).predict(X[te])
met=y==0; nm=~met
inside=(y>=lo)&(y<=hi)
def rep(mask,name):
    print(f"{name:10s} n={mask.sum():4d} coverage {inside[mask].mean():.3f}  below lo {(y<lo)[mask].mean():.3f}  above hi {(y>hi)[mask].mean():.3f}  mean width {(hi-lo)[mask].mean():.3f}")
rep(np.ones(len(y),bool),"all"); rep(met,"metals"); rep(nm,"non-metals")
print(f"metals: lo<=0 fraction {(lo[met]<=0).mean():.3f}; lo<0 {(lo[met]<0).mean():.3f}; lo exactly within 1e-6 of 0 {(np.abs(lo[met])<1e-6).mean():.3f}")
print(f"metals: median of lo {np.median(lo[met]):.4f}, median hi {np.median(hi[met]):.3f}")
print(f"non-metals: lo<=0 fraction {(lo[nm]<=0).mean():.3f}; median width {np.median((hi-lo)[nm]):.3f}")
print(f"crossing lo>hi: {(lo>hi).mean():.4f}")
# coverage with 'touching' convention: count y==lo==0 inside: already inclusive
# metals whose upper bound is below 0.05 eV (confident metal)
print(f"metals with hi<0.1 eV: {(hi[met]<0.1).mean():.3f}; non-metals with hi<0.1: {(hi[nm]<0.1).mean():.3f}")
print(f"median-model MAE {np.mean(np.abs(md-y)):.3f}; metals with median pred <0.05: {(md[met]<0.05).mean():.3f}")
print(f"time {time.time()-t0:.0f}s")
