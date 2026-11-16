import numpy as np
from scipy.optimize import least_squares
def cmd(Z,D):
    Z=np.asarray(Z,float); D=D.copy(); np.fill_diagonal(D,1.0); M=np.outer(Z,Z)/D; np.fill_diagonal(M,0.5*Z**2.4); return M
def eigs(M): return np.sort(np.linalg.eigvalsh(M))[::-1]
def D_from_R(R): return np.linalg.norm(R[:,None]-R[None],axis=-1)
rng=np.random.default_rng(0)
# Jacobian rank of eigen map wrt Cartesian coordinates
for Z in ([1,1],[8,1,1],[6,1,1,1],[6,1,1,1,1]):
    n=len(Z); R=rng.normal(0,0.8,(n,3)); h=1e-6
    f=lambda x: eigs(cmd(Z,D_from_R(x.reshape(n,3))))
    x=R.ravel(); J=np.array([(f(x+h*e)-f(x-h*e))/(2*h) for e in np.eye(3*n)]).T
    sv=np.linalg.svd(J,compute_uv=False)
    print(f"N={n}: trace(M)={np.trace(cmd(Z,D_from_R(R))):.4f} fixed; sum of eig-Jacobian rows ~ {np.abs(J.sum(0)).max():.1e}; rank={np.sum(sv>1e-6*sv[0])} of N={n}; internal dof={max(3*n-6,1)}")
# H2O-type one-parameter family: fix d_HH, solve r1, r2
Z=[8,1,1]
def Dm(r1,r2,dhh): return np.array([[0,r1,r2],[r1,0,dhh],[r2,dhh,0]],float)
r1,r2,dhh=0.96,0.98,1.52
lam0=eigs(cmd(Z,Dm(r1,r2,dhh)))
print("reference H2O-type: r1=%.3f r2=%.3f dHH=%.3f, eig="%(r1,r2,dhh),np.round(lam0,4))
for d in [1.30,1.40,1.45,1.52,1.60,1.70,1.80]:
    best=None
    for x0 in ([0.9,1.0],[1.0,0.9],[0.8,1.2],[1.2,0.8]):
        s=least_squares(lambda x: eigs(cmd(Z,Dm(x[0],x[1],d)))-lam0, x0, xtol=1e-15,ftol=1e-15,gtol=1e-15)
        a,b=sorted(s.x)
        ok = np.abs(s.fun).max()<1e-10 and b+a>d and a+d>b  # triangle inequality
        if ok: best=(a,b,np.abs(s.fun).max()); break
    print(f"  d_HH={d:.2f}:", "r_OH = %.4f, %.4f  resid %.1e"%best if best else "no valid triangle")
