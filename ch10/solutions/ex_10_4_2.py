"""Ex 10.4.2: farthest-point sampling in 1D, ties to the smaller value."""
import numpy as np
def fps(P, start, n):
    P = np.asarray(P, float); sel = [int(np.argmin(np.abs(P - start)))]
    d = np.abs(P - P[sel[0]])
    for _ in range(n - 1):
        k = int(np.flatnonzero(d == d.max())[0]); sel.append(k); d = np.minimum(d, np.abs(P - P[k]))
    return P[sel], d.max()
print("uniform 0..10, start 5:", *fps(range(11), 5, 4))
P2 = list(np.round(np.arange(0, 1.01, 0.1), 1)) + [10]
for s in [0.5, 0.0, 10]:
    sel, cover = fps(P2, s, 4); print(f"clustered set, start {s}: picks {sel}, covering radius {cover:.2f}")
