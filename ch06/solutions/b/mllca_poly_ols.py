"""Ex 6.6.5(b) supplement: degree-2 polynomial with plain least squares (alpha -> 0)."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
for inter in (0.0,0.05):
    rng=np.random.default_rng(3)
    def make(n,sr,mr):
        steps=rng.integers(sr[0],sr[1]+1,n); mw=np.exp(rng.uniform(np.log(mr[0]),np.log(mr[1]),n))
        hal=(rng.random(n)<0.25).astype(float); arom=rng.integers(0,4,n); het=rng.uniform(0,0.4,n)
        lnG=-1.0+0.30*steps+0.45*np.log(mw/100)+0.8*hal+0.10*arom+inter*steps*np.log(mw/100)+rng.normal(0,0.35,n)
        return np.column_stack([steps,np.log(mw/100),hal,arom,het]),lnG
    Xtr,ytr=make(400,(1,5),(30,300)); Xin,yin=make(400,(1,5),(30,300)); Xout,yout=make(400,(6,10),(300,800))
    m=make_pipeline(PolynomialFeatures(2,include_bias=False),LinearRegression()).fit(Xtr,ytr)
    print(inter,"poly2 OLS: in",round(np.mean(np.abs(m.predict(Xin)-yin)),3),"OOD",round(np.mean(np.abs(m.predict(Xout)-yout)),3))
