# Ex 2.5: Taylor-matching derivation of RK4 / RK2 weights on f = lambda*y
import sympy as sp
lam, h, y = sp.symbols('lambda h y')
z = lam*h
f = lam*y
# total derivatives along solution: f^(m) = lambda^(m+1) y
fp, fpp, fppp, f4 = lam**2*y, lam**3*y, lam**4*y, lam**5*y
k1 = f
k2 = lam*(y + h/2*k1)
k3 = lam*(y + h/2*k2)
k4 = lam*(y + h*k3)
print("(a) k2 - (f + h/2 f')      =", sp.simplify(k2 - (f + h/2*fp)))
print("    2(k2-k1) - h f'         =", sp.simplify(2*(k2-k1) - h*fp))
print("(b) k3 - (k2 + h^2/4 f'')   =", sp.simplify(k3 - (k2 + h**2/4*fpp)))
print("    4(k3-k2) - h^2 f''      =", sp.simplify(4*(k3-k2) - h**2*fpp))
print("(c) k4 - (f+hf'+h^2/2 f''+h^3/4 f''') =", sp.simplify(k4 - (f+h*fp+h**2/2*fpp+h**3/4*fppp)))
print("    4(k4+k1-2k3) - h^3 f''' =", sp.simplify(4*(k4+k1-2*k3) - h**3*fppp))
taylor = y + h*k1 + h*2*(k2-k1)/2 + h*4*(k3-k2)/6 + h*4*(k4+k1-2*k3)/24
rk4 = y + h/6*(k1+2*k2+2*k3+k4)
print("    Taylor-substituted - RK4 =", sp.simplify(taylor - rk4))
# weights: collect coefficients of k_i symbolically
K1,K2,K3,K4 = sp.symbols('k1:5')
expr = K1 + (K2-K1) + sp.Rational(2,3)*(K3-K2) + sp.Rational(1,6)*(K4+K1-2*K3)
print("    weights (k1..k4):", [sp.expand(expr).coeff(k) for k in (K1,K2,K3,K4)])
print("    g(z) =", sp.expand(sp.simplify(rk4/y).subs(lam, z/h)))
print("    exact Taylor e^z - g(z) leading:", sp.series(sp.exp(z)-sp.simplify(rk4/y), h, 0, 6).removeO())
# (d) RK2 midpoint
expr2 = K1 + (K2-K1)   # y + h*k1 + h^2/2 f'  with h f' = 2(k2-k1)
print("(d) RK2 weights (k1,k2):", [sp.expand(expr2).coeff(k) for k in (K1,K2)])
rk2 = y + h*k2
print("    RK2 vs Taylor to O(h^2):", sp.simplify(rk2 - (y + h*f + h**2/2*fp)))

# Caveat: nonlinear autonomous f(y)=y^2 -- identity (b) picks up extra terms
Y = sp.symbols('Y', positive=True)
F = lambda u: u**2
fp_, fpp_ = 2*Y**3, 6*Y**4          # y'=y^2 -> y''=2y^3, y'''=6y^4
k1n = F(Y); k2n = F(Y + h/2*k1n); k3n = F(Y + h/2*k2n)
print("nonlinear f=y^2: 4(k3-k2) - h^2 f'' =", sp.series(4*(k3n-k2n) - h**2*fpp_, h, 0, 3).removeO())
