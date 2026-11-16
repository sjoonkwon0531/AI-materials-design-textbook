"""Ex 5.7.1 (Metropolis acceptance table), 5.7.2 (checkerboard kernels on a 2-spin system), 5.7.3 (t_ind extrapolation)."""
import numpy as np, itertools
T = 2.4
print("5.7.1:")
for x in (-4, -2, 0, 2, 4):
    dE = 2*x
    print(f"  s_i*sum s_j = {x:+d}: dE = {dE:+d} J, acceptance = {min(1.0, np.exp(-dE/T)):.4f}")
# 5.7.2: two spins, H = -J s1 s2, beta J = 0.5; states ordered (s1,s2) = (+,+),(+,-),(-,+),(-,-)
bJ = 0.5; S = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
E = np.array([-s1*s2 for s1, s2 in S], float)          # in units of J
pi = np.exp(-bJ*E); pi /= pi.sum()
def K_site(site):
    K = np.zeros((4, 4))
    for a, s in enumerate(S):
        t = list(s); t[site] = -t[site]; b = S.index(tuple(t))
        acc = min(1.0, np.exp(-bJ*(E[b]-E[a])))
        K[a, b] = acc; K[a, a] = 1 - acc
    return K
KB, KW = K_site(0), K_site(1); D = np.diag(pi)
np.set_printoptions(precision=4, suppress=True)
print("5.7.2: pi =", pi); print("K_B =\n", KB); print("K_W =\n", KW)
for name, K in (("K_B", KB), ("K_W", KW), ("K_B K_W", KB@KW), ("K_W K_B", KW@KB), ("K_B K_W K_B", KB@KW@KB)):
    F = D@K
    print(f"  {name:12s}: stationarity |pi K - pi| = {np.abs(pi@K-pi).max():.1e}, detailed-balance violation max|DK-(DK)^T| = {np.abs(F-F.T).max():.4f}")
print("  K_B K_W =\n", KB@KW)
F = D@KB@KW; i, j = np.unravel_index(np.argmax(np.abs(F-F.T)), F.shape)
print(f"  example: pi({S[i]})K({S[i]}->{S[j]}) = {F[i,j]:.4f}  vs  pi({S[j]})K({S[j]}->{S[i]}) = {F[j,i]:.4f}")
# 5.7.3
L0, L1 = 128, 1024; tM, tW = 786.8, 9.92
for zM in (2.00, 2.17):
    tM1 = tM*(L1/L0)**(2+zM); tW1 = tW*(L1/L0)**(2+0.18)
    print(f"5.7.3: zM={zM}: Metropolis t_ind(1024) = {tM1:.3e} ms = {tM1/1e3:.0f} s = {tM1/3.6e6:.2f} h; "
          f"Wolff = {tW1:.1f} ms = {tW1/1e3:.2f} s; ratio = {tM1/tW1:.0f} (at L=128: {tM/tW:.0f})")
