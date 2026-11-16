"""Ex 1.9.B (includes the 1.9.A check): 1D Ising transfer matrix eigenvalues, free energy, magnetization m(T,h)."""
import sympy as sp, numpy as np, itertools
b, J, h, lam = sp.symbols('beta J h lambda', positive=True)
T0 = sp.Matrix([[sp.exp(b*J), sp.exp(-b*J)], [sp.exp(-b*J), sp.exp(b*J)]])
ev = sp.solve(sp.det(T0 - lam*sp.eye(2)), lam)
print("1.9.A eigenvalues:", [sp.simplify(e.rewrite(sp.cosh)) for e in ev])
print("  check:", [sp.simplify((e - 2*sp.cosh(b*J)).rewrite(sp.exp)) for e in ev], [sp.simplify((e - 2*sp.sinh(b*J)).rewrite(sp.exp)) for e in ev])
# 1.9.B general eigenvalue and magnetization
hh = sp.symbols('h', real=True)
lp = sp.exp(b*J)*sp.cosh(b*hh) + sp.sqrt(sp.exp(2*b*J)*sp.sinh(b*hh)**2 + sp.exp(-2*b*J))
T = sp.Matrix([[sp.exp(b*(J+hh)), sp.exp(-b*J)], [sp.exp(-b*J), sp.exp(b*(J-hh))]])
print("  lambda_+ is eigenvalue:", sp.simplify((T - lp*sp.eye(2)).det().rewrite(sp.exp)) == 0)
m = sp.simplify(sp.diff(sp.log(lp), hh)/b)
m_claim = sp.sinh(b*hh)/sp.sqrt(sp.sinh(b*hh)**2 + sp.exp(-4*b*J))
print("1.9.B m(T,h) matches claim:", sp.simplify((m - m_claim).rewrite(sp.exp)) == 0)
f = sp.lambdify((b, J, hh), m_claim, 'mpmath')
import mpmath as mp
mp.mp.dps = 30
for beta in [0.5, 2, 10, 50]:
    print(f"  beta={beta}: m(h=0)={f(beta,1,0)}, m(h=+0.01)={mp.nstr(f(beta,1,0.01),8)}, m(h=-0.01)={mp.nstr(f(beta,1,-0.01),8)}")
# zero-field susceptibility chi = beta e^{2 beta J}
chi = sp.simplify(sp.diff(m_claim, hh).subs(hh, 0))
print("  chi(h=0) =", chi)
# brute-force check: periodic chain N=10, compare <s>/N with exact finite-N formula d ln Tr T^N / d(beta h)
def brute(N, beta, Jv, hv):
    Z = 0; M = 0
    for s in itertools.product([-1, 1], repeat=N):
        s = np.array(s); E = -Jv*np.sum(s*np.roll(s, -1)) - hv*s.sum(); w = np.exp(-beta*E); Z += w; M += w*s.mean()
    return M/Z
def tm(N, beta, Jv, hv, d=1e-6):
    def lnZ(hv):
        Tm = np.array([[np.exp(beta*(Jv+hv)), np.exp(-beta*Jv)], [np.exp(-beta*Jv), np.exp(beta*(Jv-hv))]])
        return np.log(np.trace(np.linalg.matrix_power(Tm, N)))
    return (lnZ(hv+d)-lnZ(hv-d))/(2*d)/(beta*N)
print(f"  N=10 brute <m>={brute(10,0.7,1,0.2):.10f}, transfer-matrix={tm(10,0.7,1,0.2):.10f}, N->inf formula={float(f(0.7,1,0.2)):.10f}")
