import numpy as np
exec(open('../ch02/labs/b05_L1435.py').read())
print(f"{'N':>5} {'spectral_err':>14} {'FDM2_err':>12} {'FDM_order':>11}")
prev=None;prevN=None
for N in [4,8,12,16,24,32,48,64]:
    s=spectral_deriv(N); f=fdm_deriv(N)
    o='--' if prev is None else f"{np.log(prev/f)/np.log(N/prevN):.3f}"
    print(f"{N:5d} {s:14.3e} {f:12.3e} {o:>11}"); prev=f;prevN=N
