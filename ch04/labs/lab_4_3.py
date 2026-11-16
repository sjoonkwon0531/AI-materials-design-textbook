import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[_v]='1'  # single-thread: clean O(N^3) exponent
import numpy as np, scipy.linalg as la, time
np.random.seed(0)
# Lab 4.3  Why DFT is O(N^3): dense symmetric diagonalization cost (recovered).
sizes = [200, 400, 800, 1200, 1600, 2400, 3200]
times = []
print("=== Diagonalization wall-time vs matrix size (scipy eigvalsh) ===")
print("   N      time(s)   reps")
for N in sizes:
    A = np.random.randn(N, N); A = 0.5*(A + A.T)
    reps = 5 if N <= 800 else (3 if N <= 1600 else 1)
    t0 = time.perf_counter()
    for _ in range(reps): _ = la.eigvalsh(A)
    dt = (time.perf_counter()-t0)/reps; times.append(dt)
    print(f" {N:5d}   {dt:8.4f}   {reps}")
sizes = np.array(sizes,float); times=np.array(times)
p_full,_ = np.polyfit(np.log(sizes), np.log(times),1)
mask = sizes>=800
p_tail,_ = np.polyfit(np.log(sizes[mask]), np.log(times[mask]),1)
print(f"\nfull-range fit p = {p_full:.2f}  |  asymptotic tail (N>=800) p = {p_tail:.2f}  (ideal=3)")
print("local doubling exponents:")
for i in range(1,len(sizes)):
    r=sizes[i]/sizes[i-1]; f=times[i]/times[i-1]
    print(f"  {int(sizes[i-1])}->{int(sizes[i])}: time x{f:.2f}  -> exponent {np.log(f)/np.log(r):.2f}")
