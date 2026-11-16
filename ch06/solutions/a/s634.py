import numpy as np, warnings; warnings.filterwarnings("ignore")
from common import load_nonmetal
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, KFold
d,Xf,y,names,chemsys,nel=load_nonmetal()
print("n",len(y),"p",Xf.shape[1],"element counts:",{k:int((nel==k).sum()) for k in np.unique(nel)})
models={"ridge":make_pipeline(StandardScaler(),Ridge(alpha=1.0)),
        "1-NN":make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=1)),
        "5-NN":make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=5,weights="distance"))}
def nn_dist(cv,g):
    out=np.empty(len(y))
    for tr,te in cv.split(Xf,y,g):
        sc=StandardScaler().fit(Xf[tr]); nn=NearestNeighbors(n_neighbors=1).fit(sc.transform(Xf[tr])); out[te]=nn.kneighbors(sc.transform(Xf[te]))[0][:,0]
    return out
g=np.minimum(nel,4)
print("=== (a) leave-one-element-count-out (3 groups) ===")
for mn,m in models.items():
    p=cross_val_predict(m,Xf,y,cv=LeaveOneGroupOut(),groups=g,n_jobs=2)
    e=np.abs(p-y)
    print(f"  {mn:6s} MAE overall {e.mean():.3f}; per held-out group: "+", ".join(f"{k}-ary {e[g==k].mean():.3f}" for k in (2,3,4)))
# baseline: train mean of other groups
eb=np.empty(len(y))
for k in (2,3,4): eb[g==k]=np.abs(y[g==k]-y[g!=k].mean())
print(f"  baseline(train mean) MAE {eb.mean():.3f}; "+", ".join(f"{k}-ary {eb[g==k].mean():.3f}" for k in (2,3,4)))
print("  mean gap by group:", {k: round(y[g==k].mean(),3) for k in (2,3,4)})
nd=nn_dist(LeaveOneGroupOut(),g); print("  median NN distance:", round(np.median(nd),3), {k:round(np.median(nd[g==k]),3) for k in (2,3,4)})
print("=== (b) leave-cluster-out with k clusters ===")
Z=StandardScaler().fit_transform(Xf)
for k in (4,8,16):
    cl=KMeans(n_clusters=k,n_init=10,random_state=0).fit(Z).labels_
    row=[]
    for mn in ("ridge","5-NN","1-NN"):
        p=cross_val_predict(models[mn],Xf,y,cv=LeaveOneGroupOut(),groups=cl,n_jobs=2); row.append(np.mean(np.abs(p-y)))
    nd=nn_dist(LeaveOneGroupOut(),cl)
    print(f"  k={k:2d}: sizes min/max {np.bincount(cl).min()}/{np.bincount(cl).max()}; ridge {row[0]:.3f}  5-NN {row[1]:.3f}  1-NN {row[2]:.3f}; median NN dist {np.median(nd):.3f}; largest cluster share {np.bincount(cl).max()/len(y):.3f}")
nd=nn_dist(KFold(5,shuffle=True,random_state=0),None); print("  random 5-fold median NN dist", round(np.median(nd),3))
