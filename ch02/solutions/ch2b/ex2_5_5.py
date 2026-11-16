# Ex 2.5.5: Lax-Friedrichs: amplification factor, CFL, modified equation (numerical diffusion)
import numpy as np, sympy as sp
nu, th = sp.symbols('nu theta', real=True)
g = sp.cos(th) - sp.I*nu*sp.sin(th)        # (e^{i th}+e^{-i th})/2 - nu (e^{i th}-e^{-i th})/2
print("|g|^2 =", sp.simplify(sp.expand(g*sp.conjugate(g))), " = 1 - (1-nu^2) sin^2(theta)")

# modified equation via Taylor expansion
x, t, dx, dt, a = sp.symbols('x t dx dt a', positive=True); u = sp.Function('u')
U = lambda X, T: u(X, T)
def taylor(sx, st, order=4):
    expr = 0
    for i in range(order+1):
        for j in range(order+1-i):
            expr += sp.Derivative(u(x, t), *([x]*i + [t]*j)).doit()*(sx**i)*(st**j)/(sp.factorial(i)*sp.factorial(j)) if (i+j) > 0 else u(x, t)
    return expr
lhs = taylor(0, dt) - sp.Rational(1, 2)*(taylor(dx, 0) + taylor(-dx, 0))
rhs = -a*dt/(2*dx)*(taylor(dx, 0) - taylor(-dx, 0))
R = sp.expand((lhs - rhs)/dt)               # = u_t + a u_x - dx^2/(2dt) u_xx + dt/2 u_tt + ...
ut, ux = sp.Derivative(u(x, t), t), sp.Derivative(u(x, t), x)
print("truncation: (scheme)/dt =", sp.simplify(R.coeff(sp.Derivative(u(x, t), (x, 2)))), "* u_xx  +",
      sp.simplify(R.coeff(sp.Derivative(u(x, t), (t, 2)))), "* u_tt + ...")
# using u_tt = a^2 u_xx (leading order): diffusion coefficient
Dnum = sp.simplify(-(R.coeff(sp.Derivative(u(x, t), (x, 2))) + a**2*R.coeff(sp.Derivative(u(x, t), (t, 2)))))
print("modified eq: u_t + a u_x = D u_xx with D =", sp.factor(Dnum), " = dx^2/(2dt)(1-nu^2), nu = a dt/dx")

# numerical check: advect a Gaussian once around [0,1) periodic, N=200, a=1, nu=0.5
N = 200; h = 1/N; xc = np.arange(N)*h; a_ = 1.0
def exact_ad(D, T):               # advection-diffusion of Gaussian sigma0, periodic images
    s0 = 0.05; s2 = s0**2 + 2*D*T
    return sum(s0/np.sqrt(s2)*np.exp(-((xc - 0.5 - a_*T + k)**2)/(2*s2)) for k in [-2, -1, 0, 1])
for nu_, T in [(0.5, 1.0), (0.9, 1.0), (1.0, 1.0), (1.05, 1.0), (1.05, 10.0)]:
    dt_ = nu_*h/a_; n = int(round(T/dt_)); T = n*dt_
    uu = exact_ad(0, 0)
    with np.errstate(all='ignore'):
        for _ in range(n):
            uu = 0.5*(np.roll(uu, -1) + np.roll(uu, 1)) - nu_/2*(np.roll(uu, -1) - np.roll(uu, 1))
    Dlf = h*h/(2*dt_)*(1 - nu_**2); Dup = a_*h*(1 - nu_)/2
    if np.all(np.isfinite(uu)) and np.abs(uu).max() < 10:
        print(f"nu={nu_:4.2f} T={T:4.1f}: D_LF={Dlf:.2e} (upwind would be {Dup:.2e}); peak={uu.max():.4f}; "
              f"max err vs pure advection={np.abs(uu-exact_ad(0,T)).max():.3e}, vs modified eq={np.abs(uu-exact_ad(Dlf,T)).max():.3e}")
    else:
        print(f"nu={nu_:4.2f} T={T:4.1f}: UNSTABLE, max|u| = {np.nanmax(np.abs(uu)):.2e} after {n} steps")
