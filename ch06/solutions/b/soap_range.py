"""Ex 6.7.5(d) diagnostic: how far outside the training feature range do the x>0.5 test cells lie?"""
import numpy as np
src=open("soap_mod.py").read(); exec(src[:src.index('print("(c)')])
from sklearn.preprocessing import StandardScaler
tr=x<0.5; te=x>0.5
for k,F in (("composition",X1),("SOAP inner",S_inn),("SOAP outer",S_out),("SOAP per-centre sum",S_res)):
    sc=StandardScaler().fit(F[tr]); Zt=sc.transform(F[te]); ok=np.isfinite(Zt).all(0)
    lo,hi=F[tr].min(0),F[tr].max(0); out=((F[te]<lo-1e-12)|(F[te]>hi+1e-12)).mean()
    print(f"{k:22s} frac of test entries outside train range {out:.3f}; median over test of max|z| {np.median(np.abs(Zt[:,ok]).max(1)):.1f}")
