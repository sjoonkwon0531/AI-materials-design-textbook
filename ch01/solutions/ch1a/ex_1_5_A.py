# 연습문제 1.5.A: [[4,1],[1,3]] 고유쌍과 멱방법 5회
import numpy as np
A = np.array([[4., 1.], [1., 3.]])
l1, l2 = (7 + np.sqrt(5))/2, (7 - np.sqrt(5))/2
u1 = np.array([1, (np.sqrt(5)-1)/2]); u1 /= np.linalg.norm(u1)
print(f"lambda = {l1:.6f}, {l2:.6f};  numpy: {np.linalg.eigvalsh(A)[::-1]}")
print("u1 =", u1, " numpy:", np.linalg.eigh(A)[1][:, 1])
print(f"ratio |l2/l1| = {l2/l1:.4f}, squared = {(l2/l1)**2:.4f}")
v = np.array([1., 0.])
for k in range(1, 6):
    w = A @ v; v = w/np.linalg.norm(w); lam = v @ A @ v
    ang = np.degrees(np.arccos(min(1, abs(v @ u1))))
    print(f"k={k}: v=({v[0]:.6f},{v[1]:.6f})  RQ={lam:.8f}  |RQ-l1|={abs(lam-l1):.3e}  angle to u1={ang:.4f} deg")
