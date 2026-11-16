# Ex 4.5.5: committee (5 KRR on bootstrap subsets) -> uncertainty -> one active-learning cycle
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def U_true(r): return 4.0*(r**-12 - r**-6)
class KRR_U:
    def __init__(s, gamma=2.0, lam=1e-8): s.gamma, s.lam = gamma, lam
    def fit(s, r, y):
        s.r=r.copy(); K=np.exp(-s.gamma*(r[:,None]-r[None,:])**2)
        s.alpha=np.linalg.solve(K+s.lam*np.eye(len(r)), y); return s
    def U(s,q): return np.exp(-s.gamma*(q[:,None]-s.r[None,:])**2)@s.alpha
rng = np.random.default_rng(0)
# initial data: 30 points, deliberately sparse inside [0.92,1.1] (expensive repulsive wall rarely sampled)
r0 = np.concatenate([rng.uniform(0.92,1.1,3), rng.uniform(1.1,2.5,27)])
def committee(r, K=5, frac=0.8, seed=1):
    g=np.random.default_rng(seed); ms=[]
    for k in range(K):
        idx=g.choice(len(r), int(frac*len(r)), replace=False); ms.append(KRR_U().fit(r[idx],U_true(r[idx])))
    return ms
def stats(ms, q):
    P=np.array([m.U(q) for m in ms]); return P.mean(0), P.std(0)
q = np.linspace(0.85, 2.8, 400)
dom = (q>=0.92)&(q<=2.5)
ms0 = committee(r0); mu0, sd0 = stats(ms0, q)
def report(tag, mu, sd):
    print(f"{tag}: mean sigma in-domain [0.92,2.5] = {sd[dom].mean():.3e}, max sigma in-domain = {sd[dom].max():.3e} at r={q[dom][np.argmax(sd[dom])]:.3f}; "
          f"RMSE(mean vs true) in-domain = {np.sqrt(np.mean((mu[dom]-U_true(q[dom]))**2)):.3e}")
report("cycle 0 (30 labels)", mu0, sd0)
for lo,hi in [(0.85,0.92),(0.92,1.0),(1.0,1.1),(1.1,2.5),(2.5,2.8)]:
    m=(q>=lo)&(q<hi); print(f"   sigma mean on [{lo:.2f},{hi:.2f}) = {sd0[m].mean():.3e}")
# active learning: pick 6 pool points with largest committee sigma (pool restricted to the domain of interest)
pool = np.linspace(0.92, 2.5, 500); _, sdp = stats(ms0, pool)
new=[]
for _ in range(6):                       # greedy, avoid picking neighbours of the same peak
    j=np.argmax(sdp); new.append(pool[j]); sdp[np.abs(pool-pool[j])<0.03]=-1
new=np.array(sorted(new)); print("AL picks (r):", np.round(new,3))
r1 = np.concatenate([r0,new]); ms1 = committee(r1); mu1, sd1 = stats(ms1, q)
report("cycle 1 (36 labels, AL)", mu1, sd1)
# control: 6 random extra labels
rr = np.concatenate([r0, rng.uniform(0.92,2.5,6)]); msr=committee(rr); mur,sdr=stats(msr,q)
report("control (36 labels, random)", mur, sdr)
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
ax[0].semilogy(q,sd0,label='cycle 0'); ax[0].semilogy(q,sd1,label='cycle 1 (AL)'); ax[0].semilogy(q,sdr,':',label='random +6')
ax[0].axvspan(0.85,0.92,color='0.9'); ax[0].axvspan(2.5,2.8,color='0.9'); ax[0].set_xlabel('r'); ax[0].set_ylabel('committee std'); ax[0].legend()
ax[1].plot(q,U_true(q),'k',label='true'); ax[1].plot(q,mu1,'r--',label='committee mean'); ax[1].set_ylim(-1.5,3); ax[1].plot(new,U_true(new),'o',label='AL picks'); ax[1].legend(); ax[1].set_xlabel('r')
plt.tight_layout(); plt.savefig('ex4_5_5_committee.png',dpi=120)
