# numpy re-implementation of lab_3_3.jl (monotone-chain lower hull) for cross-check
E=[("Li",1,0,0.0),("O2",0,1,0.0),("Li2O",2,1,-2.05*3),("Li2O2",2,2,-1.58*4),("LiO2",1,2,-0.90*3)]
xO=lambda e:e[2]/(e[1]+e[2]); ef=lambda e:e[3]/(e[1]+e[2])
cr=lambda o,a,b:(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
H=[]
for p in sorted((xO(e),ef(e)) for e in E):
    while len(H)>=2 and cr(H[-2],H[-1],p)<=0: H.pop()
    H.append(p)
def eah(x,e):
    for (x1,e1),(x2,e2) in zip(H[:-1],H[1:]):
        if x1-1e-12<=x<=x2+1e-12: return e-(e1+(e2-e1)*(x-x1)/(x2-x1))
for e in E:
    x,f=xO(e),ef(e); h=eah(x,f); tag="STABLE" if h<1e-8 else f"{h:+.3f} eV/atom above hull"
    print(f"  {e[0]:<6s} x_O={x:.3f} E_form={f:+.3f} eV/atom -> {tag}")
