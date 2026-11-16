import numpy as np, warnings, time; warnings.filterwarnings("ignore")
import os; os.environ.setdefault("OMP_NUM_THREADS","2")
from common import load_nonmetal
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold
t0=time.time()
d,X,y,names,chemsys,nel=load_nonmetal(); n=len(y)
G=[0.001,0.003,0.01,0.03]; A=[0.03,0.1,0.3]
def fitpred(tr,te,g,a):
    sc=StandardScaler().fit(X[tr]); m=KernelRidge(kernel="rbf",gamma=g,alpha=a).fit(sc.transform(X[tr]),y[tr]-y[tr].mean())
    return m.predict(sc.transform(X[te]))+y[tr].mean()
outer=list(GroupKFold(5).split(X,y,chemsys))
pred=np.empty(n); chosen=[]
for k,(tr,te) in enumerate(outer):
    inner=list(GroupKFold(3).split(X[tr],y[tr],chemsys[tr]))
    sc={}
    for g in G:
        for a in A:
            e=[]
            for itr,ite in inner:
                e.append(np.abs(fitpred(tr[itr],tr[ite],g,a)-y[tr[ite]]))
            sc[(g,a)]=np.concatenate(e).mean()
    gb,ab=min(sc,key=sc.get); chosen.append((gb,ab))
    pred[te]=fitpred(tr,te,gb,ab)
    fold_mae=np.abs(pred[te]-y[te]).mean()
    # also MAE of the fixed (0.01,0.03) model on this outer fold
    fixed=np.abs(fitpred(tr,te,0.01,0.03)-y[te]).mean()
    srt=sorted(sc.items(),key=lambda kv: kv[1])
    print(f"outer fold {k}: n_test={len(te)}, chosen gamma={gb}, alpha={ab}, inner MAE={sc[(gb,ab)]:.3f} (2nd best {srt[1][0]} {srt[1][1]:.3f}); outer-fold MAE={fold_mae:.3f} (fixed 0.01/0.03: {fixed:.3f})")
print(f"nested CV outer MAE = {np.abs(pred-y).mean():.3f} eV, RMSE = {np.sqrt(np.mean((pred-y)**2)):.3f}")
print("time %.1f s"%(time.time()-t0))
