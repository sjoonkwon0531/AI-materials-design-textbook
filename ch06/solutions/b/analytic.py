import numpy as np
from scipy.stats import norm, binom, beta
print("== 6.5.1")
for n in (10,100,4601):
    q=(1-1/n)**n; print(n, f"{q:.6f}", f"distinct={n*(1-q):.1f}", f"frac={1-q:.5f}")
print("e^-1", f"{np.exp(-1):.6f}")
for B in (100,200,500):
    print("B",B,"expected OOB trees per point", f"{B*np.exp(-1):.1f}", "P(no OOB tree)", f"{(1-1/4601)**(4601*B):.2e}")
print("== 6.5.2")
for rho,s2 in ((0.064,1.071),(0.039,1.165)):
    for B in (1,10,100):
        print(rho,B,f"{rho*s2+(1-rho)*s2/B:.4f}")
    print("floor",f"{rho*s2:.4f}", "Bmin", 20*(1-rho)/rho, int(np.ceil(20*(1-rho)/rho)))
print("== 6.5.3")
for lam in (0,10):
    n1,n0=30,10; G=(n0-n1)/2; H=(n0+n1)/4
    print(lam, "G",G,"H",H,"w",-G/(H+lam))
print("ln3",np.log(3))
# Newton iterations from 0 for leaf log-odds
w=0.0
for it in range(5):
    p=1/(1+np.exp(-w)); G=10*p-30*(1-p)+0  # sum g = n0*(p-0)+n1*(p-1)
    G=10*p+30*(p-1); H=40*p*(1-p); w=w-G/H; print("newton",it+1,f"{w:.6f}")
# 0.99 leaf: h per point
print("h at p=0.99", 0.99*0.01, "points to reach H=1:", 1/(0.99*0.01))
print("== 6.5.4")
err=0.3; Z=2*np.sqrt(err*(1-err)); print("Z",Z, "M>", np.log(0.01)/np.log(Z), "Z^53",Z**53,"Z^52",Z**52)
print("== 6.6.1")
m=dict(CO2=2.0,CH4=0.010,N2O=0.0010)
for name,cf in (("AR6",dict(CO2=1,CH4=29.8,N2O=273)),("AR5",dict(CO2=1,CH4=30,N2O=265))):
    c={k:m[k]*cf[k] for k in m}; tot=sum(c.values())
    print(name, {k:round(v,4) for k,v in c.items()}, round(tot,4), {k:round(100*v/tot,2) for k,v in c.items()})
print("rel diff", (2.571-2.563)/2.571)
print("== 6.6.3")
A=np.array([[1.,-5.],[0.,1.]]); f=np.array([0,1.]); s=np.linalg.solve(A,f)
Bm=np.array([[0.9,0.2],[0.3,0.0]]) # rows: CO2, fuel(background input)
g=Bm@s; print("s",s,"g",g,"CO2 total",g[0]+0.5*g[1])
Bco=np.array([0.9+0.3*0.5,0.2]); print("aggregated", Bco@s)
A2=np.array([[0.9,-5.],[0.,1.]]); s2=np.linalg.solve(A2,f); print("self-use s",s2,"CO2",Bco@s2, "ratio",(Bco@s2)/(Bco@s))
# Leontief series check
print("series", 5*sum(0.1**k for k in range(50)))
print("== 6.7.1")
n=np.array([1255,732,914,823,877]); m_=np.array([1.47,0.07,0.61,0.60,0.26])
print("fold mean",m_.mean(),"weighted",(n*m_).sum()/n.sum(), n.sum())
print("== 6.7.2")
mu,s=1.5,0.5
for p in (0.7,0.55,0.4,0.51,0.501):
    msep=p*mu
    if p>0.5:
        u=1-1/(2*p); qn=mu+s*norm.ppf(u)
        # truncated-normal version (q restricted to >0)
        F0=norm.cdf(-mu/s); qt=mu+s*norm.ppf(F0+u*(1-F0))
    else: u=None; qn=qt=0
    print(p,"MSE",msep,"u",u,"MAE-opt(normal)",qn,"(trunc)",qt)
print("P(q<0)",norm.cdf(-3))
print("p where normal quantile hits 0:", 1/(2*(1-norm.cdf(3))))
print("== 6.7.3")
n=409; a=0.1; k=int(np.ceil((n+1)*(1-a)-1e-9)); print("k",k,"(n+1)(1-a)",(n+1)*(1-a),"k/(n+1)",k/(n+1),"upper",1-a+1/(n+1))
se=np.sqrt(0.9*0.1/401); print("se",se,"z(0.925 vs 0.9)",(0.925-0.9)/se, "z vs upper",(0.925-(0.9+1/410))/se)
# conditional coverage given calibration set ~ Beta(k, n+1-k)
sd_b=beta(k,n+1-k).std(); print("beta mean",beta(k,n+1-k).mean(),"sd",sd_b,"total sd",np.sqrt(sd_b**2+se**2),"z",(0.925-0.9)/np.sqrt(sd_b**2+se**2))
# P(test coverage >= 0.925) under exchangeability: simulate
rng=np.random.default_rng(0); C=rng.beta(k,n+1-k,200000); X=rng.binomial(401,C)/401
print("P(cov>=0.925)",np.mean(X>=0.925-1e-12), "count needed", 0.925*401)
