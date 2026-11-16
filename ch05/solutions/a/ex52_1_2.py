"""Ex 5.2.1 (Metropolis vs Barker table) and 5.2.2 (LJ tail corrections)."""
import numpy as np
print("beta dE   Metropolis   Barker   ratio")
for x in (-1, 0, 1, 3):
    m = min(1.0, np.exp(-x)); bk = 1 / (1 + np.exp(x))
    print(f"{x:5d}   {m:.4f}      {bk:.4f}   {m/bk:.3f}")
# detailed-balance check of Barker: a(x->x')/a(x'->x) = e^{-beta dE}
for x in (-1, 0.5, 3):
    print(f"Barker ratio check bdE={x}: {(1/(1+np.exp(x)))/(1/(1+np.exp(-x))):.6f} vs {np.exp(-x):.6f}")

def u_tail(rho, rc): return 8/3*np.pi*rho*(1/3*rc**-9 - rc**-3)
def p_tail(rho, rc): return 16/3*np.pi*rho**2*(2/3*rc**-9 - rc**-3)
for rho in (0.5, 0.9):
    print(f"rho={rho}: u_tail={u_tail(rho,2.5):.4f}  P_tail={p_tail(rho,2.5):.4f}")
print(f"share at rho=0.9: u_tail/(U/N) = {u_tail(0.9,2.5)/-5.0347*100:.2f} %,  |P_tail|/P = {abs(p_tail(0.9,2.5))/9.0676*100:.2f} %")
for rho in (0.5, 0.9):
    print(f"rc 2.5->3.0 at rho={rho}: u ratio {u_tail(rho,3.0)/u_tail(rho,2.5):.4f}, P ratio {p_tail(rho,3.0)/p_tail(rho,2.5):.4f}, (2.5/3)^3 = {(2.5/3)**3:.4f}")
print(f"rho=0.9, rc=3.0: u_tail={u_tail(0.9,3.0):.4f}, P_tail={p_tail(0.9,3.0):.4f}")
