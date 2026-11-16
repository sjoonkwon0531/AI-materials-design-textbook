from unmix_common import *
from scipy.ndimage import gaussian_filter1d
rng=np.random.default_rng(0)
S2,S3=phase(P2),phase(P3)
F=rng.dirichlet([1,1,1],size=150)
X=np.array([f[0]*phase(P1,0.5*f[0])+f[1]*S2+f[2]*S3 for f in F])+rng.normal(0,0.01,(150,600))
print("shift range: %.3f-%.3f deg, mean %.3f"%(0.5*F[:,0].min(),0.5*F[:,0].max(),0.5*F[:,0].mean()))
def normfrac(Sm):
    Sn=Sm/np.abs(Sm).max(1,keepdims=True); Fh=np.array([nnls(Sn.T,x)[0] for x in X]); return Fh/Fh.sum(1,keepdims=True)
for ref_shift in (0.0,0.5*F[:,0].mean()):
    S=np.array([phase(P1,ref_shift),S2,S3])
    for nm,Se in (("ICA",ica(X)),("NMF3",nmf(X)[1])):
        cc,Sm,_=match(S,Se); neg=np.maximum(np.array([-s[s<0].sum()/np.abs(s).sum() for s in Sm]),0)
        Fh=normfrac(Sm)
        print(f"ref S1 shift {ref_shift:.3f}: {nm}: corr {fmt(cc)}  neg {fmt(neg)}  fracMAE {np.mean(np.abs(Fh-F)):.3f}  (per phase {fmt(np.abs(Fh-F).mean(0))})")
# residual of 3-NMF and fit vs f1
W,H,err=nmf(X); R=X-np.clip(W@H,0,None); res=np.linalg.norm(X-W@H,axis=1)
print("NMF3 recon err %.4f; corr(per-sample residual norm, |f1-mean|) = %.3f"%(err,np.corrcoef(res,np.abs(F[:,0]-F[:,0].mean()))[0,1]))
# width of the phase-1 component vs true
_,Hm,cols=match(np.array([phase(P1,0.0),S2,S3]),H); h1=Hm[0]
i=np.argmin(abs(tt-20.0)); win=(tt>19)&(tt<21.5)
def fwhm(y):
    y=y[win]; x=tt[win]; m=y.max(); above=x[y>=m/2]; return above.max()-above.min()
print("FWHM near 20 deg: true S1 %.2f, NMF phase-1 component %.2f, peak position %.2f"%(fwhm(phase(P1)),fwhm(h1),tt[win][np.argmax(h1[win])]))
# (b) k=4,5
shifts=np.linspace(-0.2,0.8,201)
lib=np.array([phase(P1,s) for s in shifts]); dS1=np.gradient(phase(P1,0.25),tt)
for k in (4,5):
    W,H,err=nmf(X,k)
    print(f"--- NMF k={k}: recon err {err:.4f}")
    for j in range(k):
        c1=np.array([np.corrcoef(H[j],l)[0,1] for l in lib]); c2=np.corrcoef(H[j],S2)[0,1]; c3=np.corrcoef(H[j],S3)[0,1]
        cd=np.corrcoef(H[j],dS1)[0,1]
        cw=np.corrcoef(W[:,j],F[:,0])[0,1]
        print(f"   comp {j}: best corr with S1(s) {c1.max():.3f} at s={shifts[c1.argmax()]:.3f}; corr S2 {c2:.3f}, S3 {c3:.3f}; corr with dS1/d2theta {cd:+.3f}; corr(W_j, f1) {cw:+.3f}")
    # for phase-1-like comps, ratio of weights vs f1
    idx=[j for j in range(k) if max(np.corrcoef(H[j],l)[0,1] for l in lib)>0.6]
    if len(idx)>=2:
        a,b=idx[0],idx[1]; frac=W[:,b]*H[b].max()/(W[:,a]*H[a].max()+W[:,b]*H[b].max()+1e-12)
        m=F[:,0]>0.1
        print(f"   phase-1-like comps {idx}: corr(share of comp {b} among them, f1) over samples with f1>0.1 = {np.corrcoef(frac[m],F[m,0])[0,1]:+.3f}")
# (c) shift-invariant representation: Gaussian blur over 2-theta, then 3-NMF; fractions on blurred data
for sig_deg in (0.0,0.15,0.3,0.5):
    s=sig_deg/(tt[1]-tt[0])
    Xb=gaussian_filter1d(X,s,axis=1) if s>0 else X
    W,H,err=nmf(Xb)
    Sb=np.array([gaussian_filter1d(phase(P1,0.5*F[:,0].mean()),s) if s>0 else phase(P1,0.5*F[:,0].mean()),gaussian_filter1d(S2,s) if s>0 else S2,gaussian_filter1d(S3,s) if s>0 else S3])
    cc,Sm,_=match(Sb,H); Sn=Sm/np.abs(Sm).max(1,keepdims=True)
    Fh=np.array([nnls(Sn.T*1.0,x)[0] for x in Xb]); Fh/=Fh.sum(1,keepdims=True)
    # true fractions in blurred units need same normalization: rescale truth by max of blurred spectra
    sc=np.abs(Sb).max(1); Ft=F*sc; Ft/=Ft.sum(1,keepdims=True)
    print(f"blur sigma {sig_deg:.2f} deg: NMF3 corr {fmt(cc)}; fracMAE vs (rescaled) truth {np.mean(np.abs(Fh-Ft)):.3f}")
W,H,err=nmf(X); _,Hm,_=match(np.array([phase(P1,0.168),S2,S3]),H)
c1=np.array([np.corrcoef(Hm[0],l)[0,1] for l in lib]); print("NMF3 phase-1 comp: best corr with S1(s) %.3f at s=%.3f"%(c1.max(),shifts[c1.argmax()]))
avg=np.mean([phase(P1,0.5*f) for f in F[:,0]],axis=0)*1.0
wavg=(F[:,0][:,None]*np.array([phase(P1,0.5*f) for f in F[:,0]])).sum(0)
print("corr with f1-weighted average of shifted S1: %.3f"%np.corrcoef(Hm[0],wavg)[0,1])
