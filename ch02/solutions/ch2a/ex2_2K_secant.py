# Ex 2.2.K: secant shooting on the Lab 2.5 rod; residual g(s) is affine in s
import numpy as np
kap, Ta, L, T0, TL = 0.01, 20.0, 10.0, 40.0, 200.0
rhs = lambda Y: np.array([Y[1], kap*(Y[0]-Ta)])
def shoot(s, dx=0.01):
    Y = np.array([T0, s])
    for _ in range(int(round(L/dx))):
        k1=rhs(Y); k2=rhs(Y+dx/2*k1); k3=rhs(Y+dx/2*k2); k4=rhs(Y+dx*k3); Y = Y+dx/6*(k1+2*k2+2*k3+k4)
    return Y[0]
g = lambda s: shoot(s) - TL
S = [0.0, 10.0, 20.0, 30.0]; G = [g(s) for s in S]
for s, gv in zip(S, G): print(f"s={s:5.1f}  g(s)={gv:+.10f}")
print("successive slopes:", [f"{(G[i+1]-G[i])/(S[i+1]-S[i]):.12f}" for i in range(3)])
m = np.sqrt(kap)
print(f"exact slope dT(L)/ds = sinh(sqrt(kap) L)/sqrt(kap) = {np.sinh(m*L)/m:.12f}")
print(f"g(0) exact = {(T0-Ta)*np.cosh(m*L)+Ta-TL:+.10f}")
# secant from two arbitrary shots
s0, s1 = 0.0, 500.0; g0, g1 = g(s0), g(s1)
s2 = s1 - g1*(s1-s0)/(g1-g0); g2 = g(s2)
s_ex = (TL - Ta - (T0-Ta)*np.cosh(m*L))*m/np.sinh(m*L)
print(f"secant 1 step: s={s2:.9f}, g={g2:.2e}; analytic T'(0)={s_ex:.9f}")
s3 = s2 - g2*(s2-s1)/(g2-g1); print(f"secant 2nd step: s={s3:.9f}, g={g(s3):.2e}")
print(f"cost: secant 3 shots x {int(L/0.01)} RK4 steps = {3*int(L/0.01)*4} f-evals; bisection (39 iters) ~ {39*int(L/0.01)*4}")
# optional plot of g(s)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ss = np.linspace(0, 30, 7); plt.figure(figsize=(4.5, 3.5)); plt.plot(ss, [g(s) for s in ss], "o-")
    plt.axhline(0, c="k", lw=0.5); plt.xlabel("s = T'(0)"); plt.ylabel("g(s)"); plt.tight_layout(); plt.savefig("ex2_2K_residual.png", dpi=150)
except Exception as e: print("plot skipped:", e)
