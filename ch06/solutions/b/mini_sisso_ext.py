"""Ex 6.8.5 extension: also allow products of two primary features in step (iii), and check where tau-like forms rank."""
import numpy as np, itertools
exec(open("mini_sisso.py").read().split("names1=list(feat)")[0])
names1=list(feat); F1=np.column_stack([feat[k] for k in names1])
_,ui=np.unique(np.round(F1,12),axis=1,return_index=True); ui=np.sort(ui); names1=[names1[i] for i in ui]; F1=F1[:,ui]
cols=[F1]; names=list(names1)
for i,j in itertools.combinations_with_replacement(range(len(names1)),2):
    if i!=j:
        cols.append(np.c_[F1[:,i]+F1[:,j],F1[:,i]-F1[:,j],F1[:,i]*F1[:,j]]); names+=[f"{names1[i]} + {names1[j]}",f"{names1[i]} - {names1[j]}",f"({names1[i]})*({names1[j]})"]
F=np.column_stack(cols); good=np.all(np.isfinite(F),0); F=F[:,good]; names=[n for n,g in zip(names,good) if g]
print("candidates with products:",F.shape[1])
src=open("mini_sisso.py").read()
exec(src[src.index("def stump_all"):src.index("tau=d.rX")])
for name,A,B in (("oxides -> halides",oxi,hal),("halides -> oxides",hal,oxi)):
    acc,thr,sg=stump_all(F[A],y[A]); j=acc.argmax()
    tt=np.mean(predict(F[B,j],thr[j],sg[j])==y[B])
    print(f"{name}: best '{names[j]}' train {acc[j]:.3f} transfer {tt:.3f}; #tied {np.sum(acc>=acc[j]-1e-12)}")
    # a tau-like 2-term surrogate: rX/rB + nA*(rA/rB)/ln(rA/rB)  (drops -nA^2)
    k=[i for i,n in enumerate(names) if n=="rX/rB + (rA/rB)/ln(rA/rB)"]
    print("   rank of 'rX/rB + g(rho)':", int((acc>acc[k[0]]).sum())+1 if k else None)
