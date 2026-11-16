import numpy as np
chi={'Fe':1.83,'O':3.44}
for name,(a,b) in {'FeO':(1,1),'Fe2O3':(2,3),'Fe3O4':(3,4)}.items():
    x=np.array([a,b])/(a+b); c=np.array([1.83,3.44]); m=x@c; ad=x@abs(c-m)
    print(name, 'xFe=%.4f mean=%.4f avgdev=%.4f'%(x[0],m,ad), '2x(1-x)*1.61=%.4f'%(2*x[0]*x[1]*1.61))
# Hammett
import math
kH=1e-4; kN=3.63e-3; kC=4.57e-5
r1=math.log10(kN/kH)/0.78; r2=math.log10(kC/kH)/(-0.17)
print('rho NO2 %.4f  CH3 %.4f'%(r1,r2), math.log10(kN/kH), math.log10(kC/kH))
for r in (r1,r2,(r1+r2)/2): print('kCl', kH*10**(r*0.23))
# LSQ through origin with 3 pts (sigma=0 for H)
s=np.array([0,0.78,-0.17]); y=np.log10(np.array([kH,kN,kC])/kH)
print('LS rho through origin', s@y/(s@s))
A=np.c_[s,np.ones(3)]; print('LS rho, intercept', np.linalg.lstsq(A,np.log10([kH,kN,kC]),rcond=None)[0])
# 6.1.4
E=np.array([0,-0.9,32.7,453.3])
c=(E.max()+E.min())/2; print('minimax c',c,'err',E.max()-c)
print('median range', np.sort(E)[1:3], 'MAE', np.mean(abs(E-np.median(E))))
for cc in (0,10,32.7): print(cc, np.mean(abs(E-cc)))
m=E.mean(); print('mean',m,'RMSE',np.sqrt(np.mean((E-m)**2)))
E3=E[:3]; print('without sc: mean',E3.mean(),'rmse',np.sqrt(np.mean((E3-E3.mean())**2)),'minimax',(E3.max()+E3.min())/2,(E3.max()-E3.min())/2)
# 6.1.5
def S(M):
    p=np.argsort(-np.linalg.norm(M,axis=1),kind='stable'); return M[np.ix_(p,p)],p
def M(s): return np.array([[2,1+s,1],[1+s,0,2],[1,2,1]],float)
for e in [1e-1,1e-2,1e-3,1e-6]:
    Sp,pp=S(M(e)); Sm,pm=S(M(-e))
    print(e, np.linalg.norm(M(e),axis=1)**2, pp, pm, np.linalg.norm(Sp-Sm))
S0p=S(M(1e-9))[0]; S0m=S(M(-1e-9))[0]; print(S0p); print(S0m); print(S0p-S0m, np.sqrt(((S0p-S0m)**2).sum()))
# eigen check
print(np.linalg.eigvalsh(M(1e-3))-np.linalg.eigvalsh(M(-1e-3)))
