"""Ex 6.7.2: expected absolute loss of hard/soft/optimal gates under the mixture (1-p) delta(0) + p N(1.5, 0.5^2)."""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
mu,s=1.5,0.5
def Eabs_q(a): # E|Y-a| for Y~N(mu,s)
    z=(a-mu)/s; return s*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z))
def loss(a,p): return (1-p)*abs(a)+p*Eabs_q(a)
print(" p     opt     hard   soft   | E-loss: opt   hard   soft")
for p in (0.3,0.45,0.5,0.52,0.55,0.6,0.7,0.8,0.9,0.99):
    opt=mu+s*norm.ppf(1-1/(2*p)) if p>0.5 else 0.0
    opt=max(opt,0.0)
    hard=mu if p>0.5 else 0.0; soft=p*mu
    print(f"{p:5.2f}  {opt:6.3f}  {hard:5.2f}  {soft:5.3f}  |       {loss(opt,p):.3f}  {loss(hard,p):.3f}  {loss(soft,p):.3f}")
