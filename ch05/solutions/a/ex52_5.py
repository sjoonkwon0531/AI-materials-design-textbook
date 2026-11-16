"""Ex 5.2.5: Wolff in a field. (i) Wolff + Metropolis acceptance min(1, exp(-2 beta h s0 |C|));
(ii) ghost-spin Wolff (rejection-free); (iii) plain single-spin Metropolis as reference.
2D Ising, J = 1. Also: acceptance of (i) at T_c vs L for fixed h."""
import numpy as np
from numba import njit

@njit(cache=True)
def metro(L, beta, h, neq, nmeas, seed):
    np.random.seed(seed)
    s = np.ones((L, L), np.int64); M = L * L
    acc = 0.0
    for sweep in range(neq + nmeas):
        for _ in range(L * L):
            i = np.random.randint(L); j = np.random.randint(L)
            nb = s[(i+1)%L, j] + s[(i-1)%L, j] + s[i, (j+1)%L] + s[i, (j-1)%L]
            dE = 2 * s[i, j] * (nb + h)
            if dE <= 0 or np.random.random() < np.exp(-beta * dE):
                s[i, j] = -s[i, j]; M += 2 * s[i, j]
        if sweep >= neq:
            acc += M / (L * L)
    return acc / nmeas

@njit(cache=True)
def wolff_field(L, beta, h, neq, nmeas, seed):
    """Wolff cluster built with p_add, then accepted with min(1, exp(-2 beta h s0 |C|))."""
    np.random.seed(seed)
    s = np.ones((L, L), np.int64); N = L * L
    padd = 1 - np.exp(-2 * beta)
    stack = np.empty((N, 2), np.int64); clist = np.empty((N, 2), np.int64)
    inc = np.zeros((L, L), np.int64); stamp = 0
    msum = 0.0; nacc = 0; ntry = 0; csum = 0.0
    nup = 0; nupacc = 0; flipsum = 0.0
    M = N
    for t in range(neq + nmeas):
        stamp += 1
        i = np.random.randint(L); j = np.random.randint(L); s0 = s[i, j]
        inc[i, j] = stamp; stack[0, 0] = i; stack[0, 1] = j; clist[0, 0] = i; clist[0, 1] = j
        top = 1; size = 1
        while top > 0:
            top -= 1; a = stack[top, 0]; b = stack[top, 1]
            for k in range(4):
                if k == 0: x, y = (a+1) % L, b
                elif k == 1: x, y = (a-1) % L, b
                elif k == 2: x, y = a, (b+1) % L
                else: x, y = a, (b-1) % L
                if inc[x, y] != stamp and s[x, y] == s0 and np.random.random() < padd:
                    inc[x, y] = stamp; stack[top, 0] = x; stack[top, 1] = y; top += 1
                    clist[size, 0] = x; clist[size, 1] = y; size += 1
        if t >= neq:
            ntry += 1; csum += size
            if s0 == 1: nup += 1
        if np.random.random() < np.exp(-2 * beta * h * s0 * size):
            if t >= neq:
                flipsum += size
                if s0 == 1: nupacc += 1
            for c in range(size):
                s[clist[c, 0], clist[c, 1]] = -s0
            M -= 2 * s0 * size
            if t >= neq: nacc += 1
        if t >= neq:
            msum += M / N
    return msum / nmeas, nacc / ntry, csum / ntry, nupacc / max(nup, 1), flipsum / max(nacc, 1)

@njit(cache=True)
def wolff_ghost(L, beta, h, neq, nmeas, seed):
    """Ghost spin s_g coupled to every spin with strength h; plain Wolff on the extended graph.
    Physical magnetisation = s_g * sum(s) / N."""
    np.random.seed(seed)
    N = L * L
    s = np.ones(N, np.int64); sg = 1
    padd = 1 - np.exp(-2 * beta); pg = 1 - np.exp(-2 * beta * h)
    stack = np.empty(N + 1, np.int64); inc = np.zeros(N + 1, np.int64); stamp = 0
    msum = 0.0; ghost_in = 0
    for t in range(neq + nmeas):
        stamp += 1
        seed_site = np.random.randint(N + 1)                 # ghost is one more site of the graph
        s0 = sg if seed_site == N else s[seed_site]
        inc[seed_site] = stamp; stack[0] = seed_site; top = 1; g_in = seed_site == N
        while top > 0:
            top -= 1; v = stack[top]
            if v == N:                                       # ghost: try every lattice spin aligned with it
                for u in range(N):
                    if inc[u] != stamp and s[u] == s0 and np.random.random() < pg:
                        inc[u] = stamp; stack[top] = u; top += 1
            else:
                a = v // L; b = v % L
                for k in range(4):
                    if k == 0: u = ((a+1) % L) * L + b
                    elif k == 1: u = ((a-1) % L) * L + b
                    elif k == 2: u = a * L + (b+1) % L
                    else: u = a * L + (b-1) % L
                    if inc[u] != stamp and s[u] == s0 and np.random.random() < padd:
                        inc[u] = stamp; stack[top] = u; top += 1
                if inc[N] != stamp and sg == s0 and np.random.random() < pg:
                    inc[N] = stamp; stack[top] = N; top += 1; g_in = True
        for u in range(N):
            if inc[u] == stamp: s[u] = -s[u]
        if inc[N] == stamp: sg = -sg
        if t >= neq:
            msum += sg * s.sum() / N
            if g_in: ghost_in += 1
    return msum / nmeas, ghost_in / nmeas

L, T, h = 16, 2.5, 0.05
beta = 1 / T
print(f"=== <m> at L={L}, T={T}, h={h} ===")
for sd in range(3):
    m1 = metro(L, beta, h, 2000, 40000, 10 + sd)
    m2, acc, cs, _, _ = wolff_field(L, beta, h, 2000, 200000, 20 + sd)
    m3, gfrac = wolff_ghost(L, beta, h, 2000, 200000, 30 + sd)
    print(f" seed set {sd}: Metropolis {m1:.4f} | Wolff+accept {m2:.4f} (acc={acc:.3f}, <C>={cs:.1f}) | ghost Wolff {m3:.4f} (ghost in cluster {gfrac:.3f})")
print("=== acceptance of Wolff+accept at T_c, h = 0.01 and 0.05 ===")
bc = 0.5 * np.log(1 + np.sqrt(2))
for hh in (0.01, 0.05):
    for LL in (16, 32, 64, 128):
        m, acc, cs, accup, fl = wolff_field(LL, bc, hh, 2000, 20000, 7)
        print(f"  h={hh}: L={LL:3d}: acc(all)={acc:.3f}, acc(s0=+1, along h)={accup:.2e}, <|C|> proposed={cs:.1f}, <|C|> accepted={fl:.1f}, <m>={m:.3f}, beta*h*L^(15/8)={bc*hh*LL**1.875:.2f}")
print("=== h scaled as h = 0.5 L^(-15/8) (fixed scaling variable) ===")
for LL in (16, 32, 64, 128):
    hh = 0.5 * LL ** -1.875
    m, acc, cs, accup, fl = wolff_field(LL, bc, hh, 2000, 20000, 8)
    print(f"  L={LL:3d}: h={hh:.2e}, acc(all)={acc:.3f}, acc(s0=+1)={accup:.3f}, <|C|>/L^2 proposed={cs/LL**2:.3f}, accepted={fl/LL**2:.3f}")
