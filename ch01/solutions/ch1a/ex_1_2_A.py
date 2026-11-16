# 연습문제 1.2.A: FP32에서 naive sqrt(x^2+1)-x 가 0이 되기 시작하는 임계값
import numpy as np
eps32 = np.finfo(np.float32).eps
def naive32(x):
    x = np.float32(x)
    return np.sqrt(x*x + np.float32(1)) - x
# Lab의 decade 샘플
for xv in [1e2, 1e3, 1e4]:
    print(f"x={xv:.0e}: naive32={naive32(xv):.4e}")
# 정밀 스캔: 모든 float32 x in [1e3, 1e4]
xs = np.arange(1000, 10001, 1, dtype=np.float32)
vals = np.sqrt(xs*xs + np.float32(1)) - xs
zero = xs[vals == 0]
nonzero = xs[vals != 0]
print("first x with naive==0 :", zero.min())
print("last  x with naive!=0 :", nonzero.max())
# 연속 스캔 (모든 float32 사이값 포함) in [2000, 5000]
lo = np.float32(2000); hi = np.float32(5000)
a = lo.view(np.int32); b = hi.view(np.int32)
allx = np.arange(a, b+1, dtype=np.int32).view(np.float32)
v = np.sqrt(allx*allx + np.float32(1)) - allx
print("all-float32 scan: first zero at", allx[v == 0].min(), " last nonzero at", allx[v != 0].max())
print("fraction zero in [2048,4096):", np.mean(v[(allx>=2048)&(allx<4096)] == 0))
print("fraction zero in [4096,5000]:", np.mean(v[(allx>=4096)] == 0))
print("eps32 =", eps32, " eps^-1/2 =", eps32**-0.5, " sqrt(2/eps) =", np.sqrt(2/eps32))
print("x=4095 naive:", naive32(4095), " exact 1/(2x)=", 1/(2*4095))
print("x=3000 naive:", naive32(3000), " exact:", 1/(2*3000))
# 모든 float32 x에 대해 naive가 '항상' 0이 되는 지점 찾기
lo = np.float32(4096); hi = np.float32(1e5)
allx = np.arange(lo.view(np.int32), hi.view(np.int32)+1, dtype=np.int32).view(np.float32)
v = np.sqrt(allx*allx + np.float32(1)) - allx
print("last nonzero in [4096,1e5]:", allx[v != 0].max() if np.any(v!=0) else None)
nz = allx[v != 0]
print("nonzero values beyond 4096 are spurious (= ulp(x)):", np.unique((v[v!=0]/np.spacing(allx[v!=0])).round(3))[:5])
# relative error of naive (all-float32) in [1000,2896]
sel = (allx>=0)
xx = np.arange(np.float32(1000).view(np.int32), np.float32(2896).view(np.int32), 997, dtype=np.int32).view(np.float32)
vv = np.sqrt(xx*xx+np.float32(1))-xx
rel = np.abs(vv.astype(np.float64) - 1/(2*xx.astype(np.float64)))/(1/(2*xx.astype(np.float64)))
print("max rel err of naive32 on [1000,2896):", rel.max())
