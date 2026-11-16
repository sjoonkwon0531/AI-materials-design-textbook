# Run the reference lab_4_2.py logic exactly as ported in lab_4_2.jl (same loop order/prints)
import sys; sys.path.insert(0,'../ch04/labs')
import numpy as np, lab_4_2 as R
Eh,dnh,n,eps,psi=R.scf(0.5,0.3,verbose=True)
print(f"converged in {len(Eh)} iters, E={Eh[-1]:.6f}; eps={eps[:4]}; Ne={np.sum(n)*R.dx:.6f}")
for al in [1.0,0.5,0.4,0.3,0.15]:
    _,dn,*_=R.scf(4.0,al,n_iter=200,tol=0.0); hit=np.where(dn<1e-9)[0]
    print(al, hit[0]+1 if len(hit) else 'never', f"{dn[-1]:.2e}")
