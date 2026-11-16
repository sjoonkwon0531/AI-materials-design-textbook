"""Ex 10.1.3: fit t(N) = a N + b N^3 to the two DFT points and extrapolate."""
import numpy as np, json
r = json.load(open('../data/101_cost.json'))
t2, t8 = r['dft']['2']['cpu'], r['dft']['8']['cpu']
for (T2, T8, tag) in [(10.0, 77.8, 'rounded'), (t2, t8, 'raw')]:
    A = np.array([[2, 8], [8, 512]], float); a, b = np.linalg.solve(A, [T2, T8])
    print(f"[{tag}] a = {a:.5g} s/atom, b = {b:.5g} s/atom^3; share of N^3 term at N=8: {b*512/T8:.3f}")
    pd = np.log(T8 / T2) / np.log(4)
    for N in [64, 216, 1000]:
        tm = a * N + b * N ** 3; ts = T8 * (N / 8) ** pd; tc = T8 * (N / 8) ** 3
        print(f"   N={N:5d}: aN+bN^3 {tm:9.4g} s | slope {pd:.3f} model {ts:9.4g} s | cubic {tc:9.4g} s | "
              f"local exponent dlnt/dlnN of aN+bN^3 = {(a*N+3*b*N**3)/(a*N+b*N**3):.3f}")
    for N in [2, 8]:
        print(f"   local exponent at N={N}: {(a*N+3*b*N**3)/(a*N+b*N**3):.3f}")
# alternative functional forms through the same two (rounded) points
A = np.array([[2, 4], [8, 64]], float); a2, b2 = np.linalg.solve(A, [10.0, 77.8])
print(f"t = aN + bN^2: a = {a2:.4g}, b = {b2:.4g}; N=64 -> {a2*64+b2*64**2:.4g} s, N=216 -> {a2*216+b2*216**2:.4g} s")
A = np.array([[2 * np.log(2), 8], [8 * np.log(8), 512]], float); a3, b3 = np.linalg.solve(A, [10.0, 77.8])
print(f"t = aN lnN + bN^3: a = {a3:.4g}, b = {b3:.4g}; N=64 -> {a3*64*np.log(64)+b3*64**3:.4g} s")
