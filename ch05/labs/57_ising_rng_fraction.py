import numpy as np, math
from numba import njit
@njit(cache=True)
def sweeps(s,beta,n,count):
    L=s.shape[0]; w4=math.exp(-4*beta); w8=math.exp(-8*beta); c0=0; tot=0
    for _ in range(n):
        for c in range(2):
            for i in range(L):
                ip=(i+1)%L; im=(i-1)%L
                for j in range((i+c)%2,L,2):
                    x=s[i,j]*(s[ip,j]+s[im,j]+s[i,(j+1)%L]+s[i,(j-1)%L])
                    if count: 
                        tot+=1
                        if x<=0: c0+=1
                    if x<=0: s[i,j]=-s[i,j]
                    elif x==2:
                        if np.random.random()<w4: s[i,j]=-s[i,j]
                    elif np.random.random()<w8: s[i,j]=-s[i,j]
    return c0/max(tot,1)
for T in (2.4,):
    s=np.ones((64,64),np.int64); sweeps(s,1/T,5000,False); print(T, sweeps(s,1/T,5000,True))
