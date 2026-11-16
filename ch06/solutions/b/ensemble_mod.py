"""Ex 6.5.5: stacking with a fully grown tree added to the base learners; ridge with VarianceThreshold."""
import os, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb, lightgbm as lgb
warnings.filterwarnings("ignore")
L="../ch06/labs/"
df=pd.read_parquet(L+"magpie_expt_gap.parquet")
X=df.drop(columns=["formula","target"]).to_numpy(np.float32); y=df["target"].to_numpy()
cols=df.drop(columns=["formula","target"]).columns
ife=int(np.where(df.formula.values=="Fe9Co7")[0][0]) if "Fe9Co7" in set(df.formula) else None
print("Fe9Co7 index",ife)
def reg_models():
    return {"ridge":make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
            "RF":RandomForestRegressor(n_estimators=200,max_features=0.33,n_jobs=2,random_state=0),
            "XGBoost":xgb.XGBRegressor(n_estimators=600,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.5,
                                       min_child_weight=2,reg_lambda=1.0,n_jobs=2,random_state=0),
            "LightGBM":lgb.LGBMRegressor(n_estimators=600,learning_rate=0.05,num_leaves=31,subsample=0.8,subsample_freq=1,
                                         colsample_bytree=0.5,min_child_samples=10,n_jobs=2,random_state=0,verbose=-1),
            "fulltree":DecisionTreeRegressor(min_samples_leaf=1,random_state=0)}
base=["ridge","RF","XGBoost","LightGBM","fulltree"]
t0=time.time(); kf=KFold(5,shuffle=True,random_state=0)
names=base+["stack (OOF)","stack (naive)","ridge+VT"]
oof={k:np.zeros(len(y)) for k in names}; mae={k:[] for k in names}; w1=[];w2=[]; trmae={k:[] for k in base}; nvt=[]
for fold,(tr,te) in enumerate(kf.split(X)):
    Xtr,ytr,Xte=X[tr],y[tr],X[te]; fte,ftr={},{}
    for k,m in reg_models().items():
        m.fit(Xtr,ytr); fte[k]=m.predict(Xte); ftr[k]=m.predict(Xtr); oof[k][te]=fte[k]; trmae[k].append(np.mean(np.abs(ftr[k]-ytr)))
    Z=np.zeros((len(tr),len(base)))
    for itr,ite in KFold(3,shuffle=True,random_state=1).split(Xtr):
        ms=reg_models()
        for j,k in enumerate(base): Z[ite,j]=ms[k].fit(Xtr[itr],ytr[itr]).predict(Xtr[ite])
    Zte=np.column_stack([fte[k] for k in base])
    meta=LinearRegression(positive=True).fit(Z,ytr); oof["stack (OOF)"][te]=meta.predict(Zte); w1.append(np.r_[meta.intercept_,meta.coef_])
    metan=LinearRegression(positive=True).fit(np.column_stack([ftr[k] for k in base]),ytr)
    oof["stack (naive)"][te]=metan.predict(Zte); w2.append(np.r_[metan.intercept_,metan.coef_])
    vt=make_pipeline(VarianceThreshold(1e-6),StandardScaler(),Ridge(alpha=10.0)).fit(Xtr,ytr)
    oof["ridge+VT"][te]=vt.predict(Xte); kept=vt[0].get_support(); nvt.append(int((~kept).sum()))
    if fold==4 or True: print(f"  fold {fold}: VT removed {list(cols[~kept])}")
    for k in names: mae[k].append(np.mean(np.abs(oof[k][te]-y[te])))
    print(f"  fold {fold} done {time.time()-t0:.0f}s", flush=True)
print(f"{'model':14s} MAE mean  sd    fold MAEs              train MAE")
for k in names:
    print(f"{k:14s} {np.mean(mae[k]):.3f}  {np.std(mae[k],ddof=1):.3f}  "+" ".join(f"{v:.3f}" for v in mae[k])+(f"   {np.mean(trmae[k]):.3f}" if k in trmae else ""))
print("meta weights [intercept, "+", ".join(base)+"]")
print("  OOF  :",np.array2string(np.mean(w1,0),precision=3)); print("  naive:",np.array2string(np.mean(w2,0),precision=3))
print("  naive per fold:",np.array2string(np.array(w2),precision=3))
if ife is not None:
    print(f"Fe9Co7: true {y[ife]:.2f}; ridge {oof['ridge'][ife]:.2f}; ridge+VT {oof['ridge+VT'][ife]:.3f}")
print("ridge+VT max|pred|",np.abs(oof["ridge+VT"]).max())
C=np.corrcoef(np.column_stack([oof[k]-y for k in base]).T); print("OOF residual corr\n",np.array2string(C,precision=3))
print(f"total {time.time()-t0:.0f}s")
