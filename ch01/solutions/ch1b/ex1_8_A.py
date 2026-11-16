"""Ex 1.8.A: f(x)=(x^3-sin x)/(e^x+cos x), f'(1.5) four ways + FD error curve."""
import math, numpy as np, sympy as sp, torch, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
x0 = 1.5
f = lambda x: (x**3 - math.sin(x))/(math.exp(x) + math.cos(x))
def fp_hand(x):
    u, v = x**3 - math.sin(x), math.exp(x) + math.cos(x)
    du, dv = 3*x**2 - math.cos(x), math.exp(x) - math.sin(x)
    return (du*v - u*dv)/v**2
X = sp.Symbol('x'); fs = (X**3 - sp.sin(X))/(sp.exp(X) + sp.cos(X))
ref = sp.N(sp.diff(fs, X).subs(X, sp.Rational(3, 2)), 30)
fp_sym = float(ref)
xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
yt = (xt**3 - torch.sin(xt))/(torch.exp(xt) + torch.cos(xt)); yt.backward()
fp_ad = xt.grad.item()
fp_jax = float(jax.grad(lambda x: (x**3 - jnp.sin(x))/(jnp.exp(x) + jnp.cos(x)))(x0))
print(f"f(1.5) = {f(x0):.16f}")
print(f"reference (SymPy, 30 digits) = {ref}")
for name, v in [("hand", fp_hand(x0)), ("SymPy", fp_sym), ("torch AD", fp_ad), ("jax AD", fp_jax)]:
    print(f"  {name:9s} {v:.16f}  |err|={abs(v-float(ref)):.1e}")
hs = 10.0**-np.arange(1, 16)
ef = [abs((f(x0+h)-f(x0))/h - float(ref)) for h in hs]
ec = [abs((f(x0+h)-f(x0-h))/(2*h) - float(ref)) for h in hs]
for h, a, c in zip(hs, ef, ec): print(f"  h=1e{int(round(np.log10(h))):+d}: forward err={a:.2e}, central err={c:.2e}")
hf = 10.0**np.linspace(-15, -1, 1401)
ef2 = np.array([abs((f(x0+h)-f(x0))/h - float(ref)) for h in hf])
ec2 = np.array([abs((f(x0+h)-f(x0-h))/(2*h) - float(ref)) for h in hf])
print(f"best forward h ~ {hf[ef2.argmin()]:.1e} (err {ef2.min():.1e});  best central h ~ {hf[ec2.argmin()]:.1e} (err {ec2.min():.1e})")
eps = np.finfo(float).eps; print(f"theory: sqrt(eps)={eps**0.5:.1e}, eps^(1/3)={eps**(1/3):.1e}")
# plotting (optional)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.loglog(hs, ef, 'o-', label='forward'); plt.loglog(hs, ec, 's-', label='central')
    plt.xlabel('h'); plt.ylabel('|error|'); plt.legend(); plt.savefig('ex1_8_A_err.png', dpi=120)
except Exception as e: print("plot skipped:", e)
