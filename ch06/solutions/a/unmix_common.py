import numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.optimize import linear_sum_assignment, nnls
from sklearn.decomposition import PCA, FastICA, NMF
tt=np.linspace(10,70,600)
def pv(x,x0,fwhm=0.4,eta=0.5):
    g=np.exp(-4*np.log(2)*(x-x0)**2/fwhm**2); l=1/(1+4*(x-x0)**2/fwhm**2); return eta*l+(1-eta)*g
def phase(peaks,shift=0.0):
    s=sum(h*pv(tt,x0+shift) for x0,h in peaks); return s/s.max()
P1=[(20.0,1.0),(28.5,0.6),(33.0,0.8),(47.0,0.4),(56.0,0.3)]
P2=[(22.0,0.7),(31.0,1.0),(38.5,0.5),(44.5,0.6),(64.0,0.3)]
P3=[(25.5,1.0),(35.0,0.4),(41.0,0.7),(52.0,0.5),(60.0,0.4)]
def match(S_true,S_est):
    k=len(S_true); C=np.corrcoef(np.vstack([S_true,S_est]))[:k,k:]
    r,c=linear_sum_assignment(-np.abs(C))
    return np.abs(C[r,c]), np.array([np.sign(C[i,j])*S_est[j] for i,j in zip(r,c)]), c
def metrics(S,Se,X,F):
    cc,Sm,_=match(S,Se); neg=np.maximum(np.array([-s[s<0].sum()/np.abs(s).sum() for s in Sm]),0)
    Fh=np.array([nnls(Sm.T,x)[0] for x in X]); Fh/=Fh.sum(1,keepdims=True)
    return cc,neg,np.mean(np.abs(Fh-F)),Fh
def ica(X):
    m=FastICA(3,whiten="unit-variance",random_state=0,max_iter=2000,tol=1e-6).fit(X.T)
    return (X.T@m.components_.T).T
def nmf(X,k=3,init="nndsvda",rs=0):
    m=NMF(k,init=init,random_state=rs,max_iter=5000,tol=1e-6); W=m.fit_transform(np.clip(X,0,None)); return W,m.components_,m.reconstruction_err_
def fmt(v): return ", ".join(f"{x:.3f}" for x in v)
