# numpy/jax mirror of lab_2_15.jl: same network, residual, forward-mode Jacobian (jax.jacfwd ~ ForwardDiff)
# and the same hand-written Levenberg–Marquardt loop. Run over several seeds because Julia's RNG differs.
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
H=20; xc=jnp.linspace(0,1,80); f=jnp.pi**2*jnp.sin(jnp.pi*xc)
def fwd(p,x):
    w,b,v,c=p[:H],p[H:2*H],p[2*H:3*H],p[3*H]; a=jnp.tanh(jnp.outer(x,w)+b)
    return a@v+c, (-2*a*(1-a**2))@(v*w**2)
def residual(p):
    u,uxx=fwd(p,xc); u0,_=fwd(p,jnp.array([0.])); u1,_=fwd(p,jnp.array([1.]))
    return jnp.concatenate([-uxx-f, jnp.sqrt(10.)*jnp.array([u0[0],u1[0]])])
jac=jax.jit(jax.jacfwd(residual)); res=jax.jit(residual)
def lm(p, maxiter=3000, xtol=1e-10, ftol=1e-15):
    mu=1e-3; r=np.asarray(res(p)); cost=0.5*r@r; it=0
    for it in range(1,maxiter+1):
        J=np.asarray(jac(p)); g=J.T@r; A=J.T@J; d=np.diag(A); D=np.diag(np.maximum(d,1e-10*d.max()))
        while True:
            dp=-np.linalg.solve(A+mu*D, g)
            pn=p+dp; rn=np.asarray(res(pn)); cn=0.5*rn@rn
            if cn<cost: break
            mu*=4.0
            if mu>1e20: return p,cost,it
        rel=(cost-cn)/cost; p,r,cost=pn,rn,cn; mu=max(mu/3.0,1e-12)
        if np.linalg.norm(dp)<xtol*(np.linalg.norm(p)+xtol) or rel<ftol: break
    return p,cost,it
for seed in range(6):
    p0=0.5*np.random.default_rng(seed).standard_normal(3*H+1)
    p,cost,it=lm(p0)
    xf=np.linspace(0,1,1001); uf,_=fwd(p,xf); err=np.abs(np.asarray(uf)-np.sin(np.pi*xf)).max()
    print("seed",seed,"iters",it,"cost %.2e maxerr %.2e u0 %.1e u1 %.1e"%(cost,err,fwd(p,jnp.array([0.]))[0][0],fwd(p,jnp.array([1.]))[0][0]))
