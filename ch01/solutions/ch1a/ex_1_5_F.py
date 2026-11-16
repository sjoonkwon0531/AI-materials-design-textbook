# 연습문제 1.5.F: 인구 데이터 2차 회귀 — pinv / 정규방정식 / SVD 유사역행렬
import numpy as np, mpmath as mp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
yr  = np.array([1930,1940,1949,1960,1970,1980,1990,2000,2010], float)
pop = np.array([2044,2355,2017,2499,3144,3741,4339,4599,4799], float)
A = np.column_stack([yr**2, yr, np.ones_like(yr)]); B = pop
# 기준해: 50자리 정밀 정규방정식 (A가 정확히 표현되므로 참값)
mp.mp.dps = 50
Am = mp.matrix(A.tolist()); Bm = mp.matrix(B.tolist())
Xref = mp.lu_solve(Am.T*Am, Am.T*Bm); Xref = np.array([float(v) for v in Xref])
U, s, Vt = np.linalg.svd(A, full_matrices=False)
Xa = np.linalg.pinv(A) @ B
try:
    Xb = np.linalg.inv(A.T @ A) @ A.T @ B
except np.linalg.LinAlgError as e:
    Xb = None; print("normal eq. failed:", e)
Xc = Vt.T @ np.diag(1/s) @ U.T @ B
Xl = np.linalg.lstsq(A, B, rcond=None)[0]
print("singular values:", s, " kappa(A) =", f"{s[0]/s[-1]:.3e}", " kappa(A)^2 =", f"{(s[0]/s[-1])**2:.3e}")
print("cond(A^T A) by numpy:", f"{np.linalg.cond(A.T@A):.3e}")
def digits(X): return -np.log10(np.abs(X - Xref)/np.abs(Xref))
for nm, X in [("ref(50 digits)", Xref), ("(a) pinv", Xa), ("(b) normal eq", Xb), ("(c) SVD own", Xc), ("lstsq", Xl)]:
    rms = np.sqrt(np.mean((A @ X - B)**2))
    print(f"{nm:15s}: a={X[0]:.10e} b={X[1]:.10e} c={X[2]:.10e}  RMS={rms:.4f}" + ("" if nm.startswith("ref") else f"  correct digits(a,b,c)={np.round(digits(X),1)}"))
print("agreement digits (a) vs (c):", np.round(-np.log10(np.abs(Xa-Xc)/np.abs(Xc)), 1))
print("agreement digits (b) vs (c):", np.round(-np.log10(np.abs(Xb-Xc)/np.abs(Xc)), 1))
print(f"eps*kappa = {np.finfo(float).eps*s[0]/s[-1]:.2e},  eps*kappa^2 = {np.finfo(float).eps*(s[0]/s[-1])**2:.2e}")
# (d) 절단 SVD: RMS와 ||X|| vs t
print(" t   RMS        ||X||_2")
rms_t, nrm_t = [], []
for t in (1, 2, 3):
    sinv = np.array([1/si if i < t else 0 for i, si in enumerate(s)])
    X = Vt.T @ np.diag(sinv) @ U.T @ B; rms_t.append(np.sqrt(np.mean((A@X-B)**2))); nrm_t.append(np.linalg.norm(X))
    print(f" {t}  {rms_t[-1]:8.2f}   {nrm_t[-1]:.4e}")
# 중심화·스케일링 후 조건수 (보너스)
z = (yr - yr.mean())/yr.std(); Az = np.column_stack([z**2, z, np.ones_like(z)])
print(f"standardized design kappa = {np.linalg.cond(Az):.3f}")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].bar(range(1, 4), s); ax[0].set_yscale("log"); ax[0].set_xlabel("i"); ax[0].set_ylabel(r"$\sigma_i$")
ax[1].plot([1,2,3], rms_t, "o-"); ax2 = ax[1].twinx(); ax2.semilogy([1,2,3], nrm_t, "s--", c="C3")
ax[1].set_xlabel("t"); ax[1].set_ylabel("RMS"); ax2.set_ylabel("||X||")
fig.tight_layout(); fig.savefig("ex_1_5_F.png", dpi=120)
# --- 왜 정규방정식이 '전멸'하지 않았나? ---
ATA = A.T @ A; ATB = A.T @ B
ATA_exact = (Am.T*Am); ATB_exact = Am.T*Bm
print("max |A^T A| entry =", f"{np.abs(ATA).max():.3e}", " < 2^53 =", f"{2.0**53:.3e}")
print("A^T A formed exactly in FP64?", all(mp.mpf(ATA[i,j]) == ATA_exact[i,j] for i in range(3) for j in range(3)),
      "  A^T B exact?", all(mp.mpf(ATB[i]) == ATB_exact[i] for i in range(3)))
e = Xb - Xref; ce = Vt @ e; cx = Vt @ Xref
print("error components along v1,v2,v3:", ce, "  solution components:", cx)
print("relative error along v_i:", np.abs(ce/cx))
# 대조 실험: 연도를 1000으로 나눈 단위(정확히 표현 안 됨)로 같은 회귀
yk = yr/1000; Ak = np.column_stack([yk**2, yk, np.ones_like(yk)])
mp.mp.dps = 60
Akm = mp.matrix([[mp.mpf(v) for v in row] for row in Ak.tolist()])
Xk_ref = np.array([float(v) for v in mp.lu_solve(Akm.T*Akm, Akm.T*Bm)])
Xk_ne = np.linalg.solve(Ak.T @ Ak, Ak.T @ B); Xk_svd = np.linalg.pinv(Ak) @ B
print(f"[yr/1000] kappa(A)={np.linalg.cond(Ak):.3e}; digits normal eq={np.round(-np.log10(np.abs(Xk_ne-Xk_ref)/np.abs(Xk_ref)),1)}, pinv={np.round(-np.log10(np.abs(Xk_svd-Xk_ref)/np.abs(Xk_ref)),1)}")
# 열 스케일링(equilibration) 후 유효 조건수: GE/LU는 열 스케일링에 (거의) 불변
D = 1/np.linalg.norm(A, axis=0); kAD = np.linalg.cond(A*D)
print(f"kappa(A D) (columns normalized) = {kAD:.3e}, kappa(AD)^2 = {kAD**2:.3e}, eps*kappa(AD)^2 = {np.finfo(float).eps*kAD**2:.2e} -> predicted digits ~ {-np.log10(np.finfo(float).eps*kAD**2):.1f}")
print(f"eps*kappa(AD) = {np.finfo(float).eps*kAD:.2e} -> predicted digits for SVD/QR ~ {-np.log10(np.finfo(float).eps*kAD):.1f}")
