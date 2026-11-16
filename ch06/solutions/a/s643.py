from unmix_common import *
from scipy.stats import kurtosis
rng=np.random.default_rng(0); S=np.array([phase(P1),phase(P2),phase(P3)]); F=rng.dirichlet([1,1,1],size=150); X=F@S+rng.normal(0,0.01,(150,600))
Z=(S-S.mean(1,keepdims=True))/S.std(1,keepdims=True)
k=kurtosis(Z,axis=1); print("kappa_i",np.round(k,2), "corr", np.round(np.corrcoef(Z)[np.triu_indices(3,1)],3))
# pairwise 4th cross cumulant cum(s_i,s_i,s_j,s_j)=E[si^2 sj^2]-E[si^2]E[sj^2]-2E[si sj]^2
for i,j in ((0,1),(0,2),(1,2)):
    c=np.mean(Z[i]**2*Z[j]**2)-1-2*np.mean(Z[i]*Z[j])**2
    u=(Z[i]+Z[j])/np.sqrt(2); u=(u-u.mean())/u.std()
    print(f"pair {i}{j}: cum_iijj={c:.3f}; kurt of (s_i+s_j)/sqrt2 = {kurtosis(u):.2f}; independent formula (k_i+k_j)/4 = {(k[i]+k[j])/4:.2f}")
u=Z.sum(0)/np.sqrt(3); u=(u-u.mean())/u.std(); print("equal 3-mixture kurtosis %.2f; independent formula sum k/9 = %.2f"%(kurtosis(u),k.sum()/9))
# PC1 of channels-as-observations (as in the Lab)
Xc=X.T-X.T.mean(0); U,s,Vt=np.linalg.svd(Xc,full_matrices=False); pc1=U[:,0]*np.sqrt(600)
a=np.linalg.lstsq(Z.T,pc1,rcond=None)[0]; r2=1-np.sum((pc1-Z.T@a)**2)/np.sum((pc1-pc1.mean())**2)
print("PC1 = sum a_i s_i: a =",np.round(a,3),"|a|^2=%.3f R2=%.4f; kurt(PC1)=%.2f; indep formula sum a^4 k / |a|^4 = %.2f"%(a@a,r2,kurtosis(pc1),np.sum(a**4*k)/(a@a)**2))
# include cross cumulants: full 4th cumulant of sum a_i s_i
def kurt_lin(a):
    u=Z.T@a; u=(u-u.mean())/u.std(); return kurtosis(u)
print("kurt of exact linear combination with these a: %.2f"%kurt_lin(a))
