"""Ex 6.7.5 (c)(d): pooling variants on the Lab 6.18 Cu-Au cells (strain off)."""
import os, time, warnings, importlib.util
import numpy as np
L="../ch06/labs/"
spec=importlib.util.spec_from_file_location("compat",L+"67_dscribe_compat.py"); spec.loader.exec_module(importlib.util.module_from_spec(spec))
from ase.build import bulk
from ase.calculators.emt import EMT
from dscribe.descriptors import SOAP
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore"); t0=time.time()
calc=EMT()
def e_atom(at): at.calc=calc; return at.get_potential_energy()/len(at)
def make(rng,nAu,kind,strain):
    a=3.61+(4.08-3.61)*nAu/32
    at=bulk("Cu","fcc",a=a,cubic=True).repeat((2,2,2)); z=at.get_scaled_positions()[:,2]
    if kind=="random": order=rng.permutation(32)
    elif kind=="ordered": order=np.lexsort((rng.random(32),np.round(z*4).astype(int)%2))
    else: order=np.lexsort((rng.random(32),z))
    sym=np.array(["Cu"]*32); sym[order[:nAu]]="Au"
    for _ in range(int(rng.integers(0,5))):
        if 0<nAu<32:
            i=rng.choice(np.where(sym=="Cu")[0]); j=rng.choice(np.where(sym=="Au")[0]); sym[i],sym[j]="Au","Cu"
    at.set_chemical_symbols(sym)
    if strain: at.set_cell(at.cell*(1+rng.uniform(-0.02,0.02)),scale_atoms=True)
    at.rattle(0.03,seed=int(rng.integers(1e9))); return at
rng=np.random.default_rng(0); structs=[]; x=[]
for k in range(400):
    nAu=int(rng.integers(0,33)); structs.append(make(rng,nAu,("random","ordered","segregated")[k%3],False)); x.append(nAu/32)
x=np.array(x); y=np.array([e_atom(s) for s in structs])*1000
print("E/atom std %.1f meV (Lab: 18.1)"%y.std())
soap=SOAP(species=["Cu","Au"],r_cut=5.0,n_max=6,l_max=4,sigma=0.5,periodic=True)
P=soap.create(structs,n_jobs=1); Z=np.array([s.numbers for s in structs])
S_inn=SOAP(species=["Cu","Au"],r_cut=5.0,n_max=6,l_max=4,sigma=0.5,periodic=True,average="inner").create(structs,n_jobs=1)
S_out=P.mean(1)
S_res=np.concatenate([(P*(Z==29)[...,None]).sum(1),(P*(Z==79)[...,None]).sum(1)],1)/32
xAu=x[:,None]; xCu=1-xAu
S_x=np.concatenate([xAu*S_out,xCu*S_out],1)
S_x2=np.c_[S_x,x]                      # + composition column (b_Cu x_Cu + b_Au x_Au is affine in x)
S_res2=np.c_[S_res,x]
X1=np.c_[x,x**2,x**3]
ridge=lambda: make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-6,3,19)))
cv=KFold(5,shuffle=True,random_state=0)
feats={"composition (x,x^2,x^3)":X1,"SOAP inner":S_inn,"SOAP outer":S_out,"SOAP outer x [x_Au, x_Cu]":S_x,
       "SOAP outer x [x_Au, x_Cu] + x":S_x2,"SOAP per-centre sum":S_res,"SOAP per-centre sum + x":S_res2}
print("(c) 5-fold CV MAE (meV/atom):")
for k,F in feats.items():
    print(f"   {k:32s} {mean_absolute_error(y,cross_val_predict(ridge(),F,y,cv=cv)):.3f}  (dim {F.shape[1]})")
# how different is x_Au*outer from the per-centre Au sum? relative difference
Au_sum=(P*(Z==79)[...,None]).sum(1)/32; diff=np.linalg.norm(xAu*S_out-Au_sum,axis=1)/np.linalg.norm(Au_sum+1e-30,axis=1)
m=(x>0)&(x<1); print("   rel. diff ||x_Au*outer - (1/N)sum_Au p_i|| / ||...||: median %.3f over 0<x<1 cells"%np.median(diff[m]))
# (a) numerical check of the identity: mean of p_i (as c_i c_i^*) minus outer product of mean c
print("(d) extrapolation: train x<0.5, test x>0.5")
tr=x<0.5; te=x>0.5
print("   n_train %d n_test %d; y train mean %.1f test mean %.1f"%(tr.sum(),te.sum(),y[tr].mean(),y[te].mean()))
for k,F in feats.items():
    m_=ridge().fit(F[tr],y[tr]); p=m_.predict(F[te])
    print(f"   {k:32s} MAE {mean_absolute_error(y[te],p):7.2f}  mean signed {np.mean(p-y[te]):+7.2f}")
print("   baseline (train mean)            MAE %7.2f"%mean_absolute_error(y[te],np.full(te.sum(),y[tr].mean())))
# reverse direction as a check
tr2,te2=x>0.5,x<0.5
print("(d') reverse: train x>0.5, test x<0.5")
for k in ("composition (x,x^2,x^3)","SOAP inner","SOAP outer","SOAP per-centre sum"):
    F=feats[k]; p=ridge().fit(F[tr2],y[tr2]).predict(F[te2]); print(f"   {k:32s} MAE {mean_absolute_error(y[te2],p):7.2f}")
# interpolation reference with same sizes: random split
rng=np.random.default_rng(5); idx=rng.permutation(400); a,b=idx[:tr.sum()],idx[tr.sum():]
print("(d'') random split of same size (reference)")
for k in ("composition (x,x^2,x^3)","SOAP inner","SOAP outer","SOAP per-centre sum"):
    F=feats[k]; p=ridge().fit(F[a],y[a]).predict(F[b]); print(f"   {k:32s} MAE {mean_absolute_error(y[b],p):7.2f}")
# per-arrangement error for composition model in extrapolation
print("time %.0f s"%(time.time()-t0))
