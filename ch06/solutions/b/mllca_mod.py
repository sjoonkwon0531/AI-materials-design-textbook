"""Ex 6.6.5: variations of the SYNTHETIC ML-LCA extrapolation lab (Lab 6.15)."""
import numpy as np, warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
warnings.filterwarnings("ignore")
def experiment(train_steps,train_mw,inter=0.0,label="",seed=3):
    rng=np.random.default_rng(seed)
    def make(n,sr,mr):
        steps=rng.integers(sr[0],sr[1]+1,n); mw=np.exp(rng.uniform(np.log(mr[0]),np.log(mr[1]),n))
        hal=(rng.random(n)<0.25).astype(float); arom=rng.integers(0,4,n); het=rng.uniform(0,0.4,n)
        lnG=-1.0+0.30*steps+0.45*np.log(mw/100)+0.8*hal+0.10*arom+inter*steps*np.log(mw/100)+rng.normal(0,0.35,n)
        return np.column_stack([steps,np.log(mw/100),hal,arom,het]),lnG
    Xtr,ytr=make(400,train_steps,train_mw); Xin,yin=make(400,train_steps,train_mw); Xout,yout=make(400,(6,10),(300,800))
    models={"linear":LinearRegression(),
            "RF":RandomForestRegressor(n_estimators=300,min_samples_leaf=2,n_jobs=2,random_state=0),
            "LightGBM":lgb.LGBMRegressor(n_estimators=400,learning_rate=0.05,num_leaves=15,n_jobs=1,random_state=0,verbose=-1),
            "poly2+ridge":make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))),
            "poly2+ridge(a=1)":make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=1.0))}
    print(f"=== {label}: train steps {train_steps}, MW {train_mw}, interaction {inter}")
    print(f"   ytr max {ytr.max():.2f}, OOD y mean {yout.mean():.2f}")
    for k,m in models.items():
        m.fit(Xtr,ytr); pi,po=m.predict(Xin),m.predict(Xout)
        extra=""
        if k=="poly2+ridge": extra=f" alpha={m[-1].alpha_:.3g}"
        print(f"   {k:17s} MAE in {np.mean(np.abs(pi-yin)):.3f}  OOD {np.mean(np.abs(po-yout)):.3f}  median ratio {np.exp(np.median(po-yout)):.3f}  OOD pred max {po.max():.2f}{extra}")
    sc=StandardScaler().fit(Xtr); nn=NearestNeighbors(n_neighbors=5).fit(sc.transform(Xtr))
    dtr=nn.kneighbors(sc.transform(Xtr),n_neighbors=6)[0][:,1:].mean(1); thr=np.percentile(dtr,95)
    d_in=nn.kneighbors(sc.transform(Xin))[0].mean(1); d_out=nn.kneighbors(sc.transform(Xout))[0].mean(1)
    print(f"   AD thr {thr:.3f}: flagged in {np.mean(d_in>thr):.3f}, OOD {np.mean(d_out>thr):.3f}")
    # overlap region of OOD with training
    ov=(Xout[:,0]<=train_steps[1])&(np.exp(Xout[:,1])*100<=train_mw[1])
    rf=models["RF"]; e=np.abs(rf.predict(Xout)-yout)
    print(f"   OOD points inside the new training box (steps<= {train_steps[1]}, MW<= {train_mw[1]}): {ov.mean():.3f}; RF MAE there {e[ov].mean() if ov.any() else float('nan'):.3f}, outside {e[~ov].mean():.3f}")
    fl=d_out>thr; print(f"   OOD RF MAE flagged {e[fl].mean() if fl.any() else float('nan'):.3f} (n={fl.sum()}), unflagged {e[~fl].mean() if (~fl).any() else float('nan'):.3f} (n={(~fl).sum()})")
experiment((1,5),(30,300),label="baseline (Lab 6.15)")
experiment((1,7),(30,450),label="(a) wider training")
experiment((1,5),(30,300),inter=0.05,label="(c) interaction 0.05*steps*ln(MW/100), original training")
experiment((1,7),(30,450),inter=0.05,label="(c') interaction, wider training")
