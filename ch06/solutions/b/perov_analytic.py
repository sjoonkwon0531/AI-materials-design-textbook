import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
L="../../../ch06/labs/"
def tf(rA,rB,rX,nA):
    t=(rA+rX)/(np.sqrt(2)*(rB+rX)); mu=rB/rX; tau=rX/rB-nA*(nA-(rA/rB)/np.log(rA/rB)); return t,mu,tau
P=lambda tau:1/(1+np.exp(-(-1.968*tau+8.348)))
print("== 6.8.1")
for name,rB,rX in (("CsPbBr3",1.19,1.96),("CsPbCl3",1.19,1.81),("CsPbI3",1.19,2.20),("Cs2AgBiBr6",(1.15+1.03)/2,1.96),("Cs2AgBiCl6",1.09,1.81)):
    t,mu,tau=tf(1.88,rB,rX,1)
    print(f"{name:11s} rB={rB:.3f} t={t:.4f} mu={mu:.4f} tau={tau:.4f} t-win={0.825<t<1.059} mu-win={0.414<mu<0.732} tau<4.18={tau<4.18} P={P(tau):.4f}")
s1=pd.read_csv(L+"bartel_TableS1.csv")
print(s1[s1.ABX3.isin(["CsPbBr3","CsPbCl3","CsPbI3"])][["ABX3","exp_label","t","tau","tau_prob"]])
s3=pd.read_csv(L+"bartel_TableS3.csv"); print(s3[s3.compound=="Cs2AgBiCl6"][["compound","t","tau","tau_prob"]])
print("== 6.8.2")
e=np.e; print("g(e)",e)
for nA,mu in ((1,0.541),(1,1.19/2.20)):
    print("bound",nA,mu,1/mu-nA**2+nA*e)
print("table:")
for nA in (1,2,3):
    print(nA,[round(1/mu-nA**2+nA*e,3) for mu in (0.4,0.5,0.6)])
tc=(8.348-np.log(4))/1.968; print("tau for P>0.8:",tc, "P at tc", P(tc))
mu_min=1/(tc+1-e); print("n_A=1: mu >",mu_min)
# Cs2AgBiCl6
rA,rB,rX=1.88,1.09,1.81; rho=rA/rB; print("rho",rho,"g",rho/np.log(rho),"1/mu",rX/rB,"tau",rX/rB-1+rho/np.log(rho),"bound",rX/rB-1+e)
# minimum over rB at fixed rA=1.88, rX=1.81 (Cl) and MA/Br
for rA,rX,lab in ((1.88,1.81,"Cs,Cl"),(2.17,1.96,"MA,Br")):
    f=lambda rB: rX/rB-1+(rA/rB)/np.log(rA/rB)
    r=minimize_scalar(f,bounds=(0.3,rA-0.01),method="bounded")
    print(lab,"min over rB: rB*",round(r.x,4),"tau_min",round(r.fun,4),"rho*",round(rA/r.x,4),"P",round(P(r.fun),4))
    sub=s3[(s3["rA (Ang)"]==rA)&(s3["rX (Ang)"]==rX)]
    i=sub.tau.idxmin(); print("   S3 min tau",sub.loc[i,["compound","rB (Ang)","tau","tau_prob"]].to_dict(), "n",len(sub))
    # tau with rB in table range
    print("   S3 rB range",sub["rB (Ang)"].min(),sub["rB (Ang)"].max())
print("S1 min tau overall",s1.tau.min(), s1.loc[s1.tau.idxmin(),"ABX3"])
