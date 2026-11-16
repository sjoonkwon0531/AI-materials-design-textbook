"""85_reach.py -- numbers for Estimate Box on the input-space reach of the SAM radius (seed-0 surrogates of Lab 8.5b)."""
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib,sys,numpy as np,torch,io,contextlib
sys.path.insert(0,'.')
src=open('85_sam_surrogate.py').read().split('# ---------------------------------------------------------------- data')[0]
exec(src)
rng = np.random.default_rng(0); n_tr=400
base = rng.uniform(650, 720, n_tr)[:, None] + rng.normal(0, 10, (n_tr, P.N)); X=(base-700)/100
y = np.array([E.rollout_batch(lambda t, o, s=s: np.full(1, s[t]), 1, 10_000 + i, P)[0][0] for i, s in enumerate(X)])
yn=(y-y.mean())/y.std()
print('median ||x||', np.median(np.linalg.norm(X,axis=1)).round(3))
for mode,rho in [('adamw',0),('sam',0.05),('sam',0.2)]:
    net=train(X,yn,0,mode,rho) if mode!='adamw' else train(X,yn,0)
    W=net[0].weight.detach().numpy(); print(mode,rho,'||W1||_op',np.linalg.norm(W,2).round(3),'||W1||_F',np.linalg.norm(W).round(3))
    xn = np.median(np.linalg.norm(X, axis=1))
    print(f"   input-space reach rho*||x||/||W1||_op = {rho * xn / np.linalg.norm(W, 2):.4f} action units = {100 * rho * xn / np.linalg.norm(W, 2):.2f} K")
