# Ex 2.3.A: tension lambda from differentiating the pendulum constraint twice (sympy)
import sympy as sp
t, g = sp.symbols('t g', positive=True)
lam = sp.Symbol('lam')
x, y = sp.Function('x')(t), sp.Function('y')(t)
phi = x**2 + y**2 - 1
dphi = sp.diff(phi, t)
ddphi = sp.diff(phi, t, 2)
# substitute equations of motion xdd = -lam x, ydd = -lam y - g
ddphi_s = ddphi.subs({sp.diff(x, t, 2): -lam*x, sp.diff(y, t, 2): -lam*y - g})
sol = sp.solve(sp.Eq(ddphi_s, 0), lam)[0]
print("phi_dot  =", sp.simplify(dphi))
print("phi_ddot =", sp.expand(ddphi_s))
print("lambda   =", sp.simplify(sol))
# on the manifold x^2+y^2=1 : lambda = v^2 - g y ; check at bottom with speed v: lam = v^2 + g
vx, vy, X, Y = sp.symbols('vx vy X Y')
lam_m = sol.subs({sp.diff(x, t): vx, sp.diff(y, t): vy, x: X, y: Y})
print("lambda(X,Y,vx,vy) =", sp.simplify(lam_m))
print("at bottom (0,-1), speed v:", sp.simplify(lam_m.subs({X: 0, Y: -1, vx: sp.Symbol('v'), vy: 0})))
# energy check: released from (1,0) at rest -> at bottom v^2 = 2 g  -> lambda = 3 g (= 3 m g tension)
print("released from horizontal, at bottom: lambda =", sp.simplify(lam_m.subs({X: 0, Y: -1, vx: sp.sqrt(2*g), vy: 0})))
print("numeric (g=9.81):", float((3*g).subs(g, 9.81)))
