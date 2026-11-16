# Cross-check of lab_4_5.jl's KRR + scalar force formula + histogram binning against lab_4_5.py's vectorized versions
import numpy as np
np.random.seed(2)
U_true=lambda r:4*(r**-12-r**-6); F_true=lambda r:48*r**-13-24*r**-7
r=np.random.uniform(0.92,2.5,160); y=U_true(r); g=2.0
alpha=np.linalg.solve(np.exp(-g*(r[:,None]-r[None,:])**2)+1e-8*np.eye(160),y)
rg=np.linspace(0.95,2.5,400)
# python (vectorized, as in lab_4_5.py)
d=rg[:,None]-r[None,:]; K=np.exp(-g*d**2); Fv=-((K*(-2*g*d))@alpha); Uv=K@alpha
# julia-style scalar loop
Fs=np.array([-sum(alpha[k]*np.exp(-g*(q-r[k])**2)*(-2*g*(q-r[k])) for k in range(160)) for q in rg])
Us=np.array([sum(alpha[k]*np.exp(-g*(q-r[k])**2) for k in range(160)) for q in rg])
print("max |dF|, |dU| between formulations:", np.abs(Fs-Fv).max(), np.abs(Us-Uv).max())
print(f"U-RMSE={np.sqrt(np.mean((Uv-U_true(rg))**2)):.2e} F-RMSE={np.sqrt(np.mean((Fv-F_true(rg))**2)):.2e} (one random draw)")
# binning check
rmax=4.78; dd=np.random.uniform(0,rmax,100000); h,_=np.histogram(dd,bins=100,range=(0,rmax))
b=np.minimum(np.floor(dd/rmax*100).astype(int),99); h2=np.bincount(b,minlength=100)
print("histogram binning identical:", np.array_equal(h,h2))
