import numpy as np, warnings, time; warnings.filterwarnings("ignore")
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
t0=time.time()
lab=np.repeat([0,1,2],300)
def make(sdB=4.0,posC=120.0):
    r=np.random.default_rng(1)
    A=r.normal(0,1.0,(300,10)); B=r.normal(0,sdB,(300,10)); C=r.normal(0,1.0,(300,10))
    B[:,0]+=30.0; C[:,0]+=posC; return np.vstack([A,B,C])
def summ(E):
    c=[E[lab==i].mean(0) for i in range(3)]
    rad=[np.sqrt(((E[lab==i]-c[i])**2).sum(1)).mean() for i in range(3)]
    return rad[1]/rad[0], np.linalg.norm(c[2]-c[0])/np.linalg.norm(c[1]-c[0])
ts=lambda S,perp=30: TSNE(2,perplexity=perp,init="pca",random_state=0,n_jobs=2).fit_transform(S)
um=lambda S: umap.UMAP(n_neighbors=15,random_state=0).fit_transform(S)
print("(a) sd(B) sweep, C at 120:  input / t-SNE30 / UMAP15  radius(B)/radius(A)   [dist ratio]")
for sd in (1,2,4,8):
    S=make(sd,120); a=summ(S); b=summ(ts(S)); c=summ(um(S))
    print(f"  sd={sd}: radius {a[0]:.2f} / {b[0]:.2f} / {c[0]:.2f}    dist {a[1]:.2f} / {b[1]:.2f} / {c[1]:.2f}")
print("(b) C position sweep, sd(B)=4: input / t-SNE30 / UMAP15 / t-SNE300  dist(A,C)/dist(A,B)")
for pc in (60,120,240):
    S=make(4,pc); a=summ(S); b=summ(ts(S)); c=summ(um(S)); e=summ(ts(S,300))
    print(f"  C at {pc}: dist {a[1]:.2f} / {b[1]:.2f} / {c[1]:.2f} / {e[1]:.2f};  radius ratio t-SNE300 {e[0]:.2f}")
print("time %.1f"%(time.time()-t0))
