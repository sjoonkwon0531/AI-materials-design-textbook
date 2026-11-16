# Cross-check of lab_1_2.jl logic in NumPy (Julia semantics mirrored where they differ)
import numpy as np
def find_eps(T):
    e = T(1.0)
    while T(1.0) + e / T(2) > T(1.0):
        e = e / T(2)
    return e
print("--- Machine epsilon (measured) ---")
for n, T in (("FP64", np.float64), ("FP32", np.float32), ("FP16", np.float16)):
    print(f"{n}: {float(find_eps(T)):.4e}")
print(f"BF16: {2.0**-7:.4e}")
print("\n--- Catastrophic cancellation ---")
for xv in (1e2, 1e4, 1e6, 1e8, 1e10):
    x32 = np.float32(xv); x64 = xv
    n32 = np.sqrt(x32**2 + np.float32(1)) - x32      # Julia: Float32 + Int -> Float32
    n64 = np.sqrt(x64**2 + 1) - x64
    rb = 1.0 / (np.sqrt(x64**2 + 1) + x64)
    print(f"x={xv:.0e}: naive(FP32)={float(n32):.4e}, naive(FP64)={n64:.4e}, robust(FP64)={rb:.4e}")
print("\n--- Sum is not associative in FP ---")
v = np.full(10**7, 1e-8, dtype=np.float32)
def pairwise(a, lo, hi, bs=1024):        # Julia Base.mapreduce_impl: pairwise, blocksize 1024
    if hi - lo < bs:
        s = np.float32(0)
        for t in a[lo:hi].tolist():      # plain sequential base case (Julia's @simd may reorder lanes)
            s = np.float32(s + np.float32(t))
        return s
    m = lo + (hi - lo) // 2
    return np.float32(pairwise(a, lo, m) + pairwise(a, m, hi))
S = pairwise(v, 0, len(v))
a = np.float32(1) + S; b = S + np.float32(1)
c = np.float32(1)
# sequential: 1 + 1e-8 rounds back to 1 in FP32 every time
for _ in range(1000): c = np.float32(c + np.float32(1e-8))
print(f"Vectorized sum:    {float(a):.10f}")
print(f"Commuted order:    {float(b):.10f}")
print(f"Sequential sum:    {float(c):.10f}   (first 1000 steps; stays 1 thereafter)")
