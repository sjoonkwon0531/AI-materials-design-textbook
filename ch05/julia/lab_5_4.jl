# Lab 5.4 (Julia) — Gillespie SSA 대 ODE: 평균, 요동, stiffness, 쌍안정
# 파이썬 labs/53_ssa.py 대응.
#  Part A: Ch.2 Lab 2.3 선형 네트워크, SSA 평균 대 행렬지수 exp(A t), 요동 대 이항 이론
#  Part B: stiffness가 SSA에서는 사건 수로 나타남 (BDF 비교는 OrdinaryDiffEq.jl의 FBDF)
#  Part C: Schlögl 쌍안정 모형, ODE(고정 간격 RK4)와 SSA 비교
# SSA 숫자는 난수 때문에 통계적으로만 일치한다. 행렬지수, 고정점, ODE 끝값은 결정론적이다.
using Printf, Random, Statistics, LinearAlgebra
using OrdinaryDiffEq          # Part B의 BDF 함수 평가 수 비교에만 사용

# ------------------------------------------------------------------ SSA 핵심
# 반응: R1: X1->X2 (k1), R2: X2->X3 (k2), R3: X3->X2 (k3), R4: X2->X1 (k4)
const NU = [-1 1 0; 0 -1 1; 0 1 -1; 1 -1 0]          # 화학량론 (행 = 반응)

"직접법 SSA (Gillespie 1976). t_obs 시각의 상태(행 = 시각)와 사건 수"
function ssa_linear(x0, k, t_obs, seed)
    Random.seed!(seed)
    x = copy(x0); a = zeros(4); out = zeros(length(t_obs), 3)
    t = 0.0; j = 1; nev = 0
    while j <= length(t_obs)
        a[1] = k[1] * x[1]; a[2] = k[2] * x[2]; a[3] = k[3] * x[3]; a[4] = k[4] * x[2]
        R = sum(a)
        tau = R <= 0.0 ? 1e300 : -log(rand()) / R          # Delta t = -ln u / R_tot
        while j <= length(t_obs) && t + tau > t_obs[j]     # 다음 사건 전에 오는 관측 시각 기록
            out[j, :] .= x; j += 1
        end
        j > length(t_obs) && break
        r = rand() * R; mu = 1; c = a[1]                    # 확률 a_mu / R로 반응 선택
        while c < r
            mu += 1; c += a[mu]
        end
        x .+= NU[mu, :]; t += tau; nev += 1
    end
    return out, nev
end

"Schlögl 모형, 고전 매개변수에 부피 인자 s (분자 수가 작아지고 상대 요동이 커짐)"
function ssa_schlogl(x0, tmax, dt_rec, seed; s = 4.0)
    Random.seed!(seed)
    c1, c2, c3, c4 = 3e-7 * s, 1e-4 * s * s, 1e-3 / s, 3.5
    N1, N2 = 1e5, 2e5
    x = x0; t = 0.0; nrec = floor(Int, tmax / dt_rec) + 1; rec = zeros(nrec); j = 0; nev = 0
    while j < nrec
        a1 = c1 * N1 * x * (x - 1) / 2.0; a2 = c2 * x * (x - 1) * (x - 2) / 6.0
        a3 = c3 * N2; a4 = c4 * x
        R = a1 + a2 + a3 + a4
        tau = -log(rand()) / R
        while j < nrec && t + tau > j * dt_rec
            rec[j+1] = x; j += 1
        end
        r = rand() * R
        if r < a1
            x += 1
        elseif r < a1 + a2
            x -= 1
        elseif r < a1 + a2 + a3
            x += 1
        else
            x -= 1
        end
        t += tau; nev += 1
    end
    return rec, nev
end

"고정 간격 RK4 (스칼라 ODE)"
function rk4(f, x0, T; h = 1e-3)
    x = x0
    for _ in 1:round(Int, T / h)
        k1 = f(x); k2 = f(x + 0.5h * k1); k3 = f(x + 0.5h * k2); k4 = f(x + h * k3)
        x += h / 6 * (k1 + 2k2 + 2k3 + k4)
    end
    return x
end

"삼차 다항식 c[1] x^3 + c[2] x^2 + c[3] x + c[4]의 근 (동반행렬 고유값, np.roots와 같은 방법)"
function cubic_roots(c)
    a = c ./ c[1]
    C = [-a[2] -a[3] -a[4]; 1.0 0.0 0.0; 0.0 1.0 0.0]
    return sort(real.(eigvals(C)))
end

function main()
    # ------------------------------------------------------------ Part A
    println("=== Part A: Ch.2 network X1 -(1001.5)-> X2 <-(0.5/1.0)-> X3 ===")
    k = [1001.5, 0.5, 1.0, 0.0]
    A = [-k[1] 0 0; k[1] -k[2] k[3]; 0 k[2] -k[3]]
    t_obs = [0.01, 0.5, 1.0, 2.0, 4.0, 8.0]; N0 = 1000; R = 400
    X = zeros(R, length(t_obs), 3); nevs = zeros(R)
    for r in 1:R
        X[r, :, :], nevs[r] = ssa_linear([N0, 0, 0], k, t_obs, 1000 + r - 1)
    end
    mn = dropdims(mean(X; dims = 1); dims = 1) ./ N0
    sem = dropdims(std(X; dims = 1); dims = 1) ./ N0 ./ sqrt(R)      # std: ddof = 1
    @printf("N0=%d, trajectories=%d, mean #events per trajectory=%.0f\n", N0, R, mean(nevs))
    println("   t    <x3>_SSA          exact(expm)   |diff|/sem")
    for (i, t) in enumerate(t_obs)
        ex = (exp(A * t) * [1.0, 0, 0])[3]                             # 행렬지수 (LinearAlgebra.exp)
        @printf("%5.2f  %.4f +/- %.4f   %.4f       %.2f\n", t, mn[i, 3], sem[i, 3], ex, abs(mn[i, 3] - ex) / max(sem[i, 3], 1e-12))
    end
    println("fluctuations of x3 = X3/N at t=8 (stationary binomial theory sqrt(p(1-p)/N), p=1/3):")
    for N in (100, 1000, 10000)
        vals = [ssa_linear([N, 0, 0], k, [8.0], 5000 + r)[1][1, 3] / N for r in 0:399]
        @printf("  N=%6d: mean=%.4f  std=%.4f  theory=%.4f\n", N, mean(vals), std(vals), sqrt((1 / 3) * (2 / 3) / N))
    end

    # ------------------------------------------------------------ Part B
    println("=== Part B: stiffness as event count (X1 <-> X2 fast, X2 <-> X3 slow) ===")
    for kfast in (10.0, 100.0, 1000.0)
        kB = [kfast, 0.5, 1.0, kfast]
        AB = [-kB[1] kB[4] 0; kB[1] -kB[2]-kB[4] kB[3]; 0 kB[2] -kB[3]]
        ev = abs.(eigvals(AB)); nz = sort(ev[ev .> 1e-9]); S = nz[end] / nz[1]
        ne = [ssa_linear([1000, 0, 0], kB, [8.0], 7000 + r)[2] for r in 0:19]
        prob = ODEProblem((du, u, p, t) -> mul!(du, AB, u), [1.0, 0.0, 0.0], (0.0, 8.0))
        sol = solve(prob, FBDF(); reltol = 1e-6, abstol = 1e-9)
        @printf("  k_fast=%7.1f: stiffness S=%8.1f  SSA events/traj=%9.0f  BDF f-evals=%4d\n", kfast, S, mean(ne), sol.stats.nf)
    end

    # ------------------------------------------------------------ Part C
    println("=== Part C: Schloegl model (bistable), volume factor s=4 ===")
    sc = 4.0
    roots = cubic_roots([-1e-4 * sc * sc / 6, 3e-7 * sc * 1e5 / 2, -3.5, 1e-3 * 2e5 / sc])
    println("  deterministic fixed points: ", round.(roots; digits = 1), " (low/high stable, middle unstable)")
    rhs(x) = 3e-7 * sc * 1e5 / 2 * x * x - 1e-4 * sc * sc / 6 * x^3 + 2e2 / sc - 3.5 * x
    for x0 in (roots[2] - 2, roots[2] + 2)
        @printf("  ODE from x0=%.1f: x(50)=%.1f\n", x0, rk4(rhs, x0, 50.0))
    end
    tmax, dtr = 2000.0, 0.05
    ssa_schlogl(60.0, 1.0, dtr, 1; s = sc)                              # JIT 예열
    t0 = time(); rec, nev = ssa_schlogl(62.0, tmax, dtr, 42; s = sc); el = time() - t0
    state = rec[1] < roots[2] ? 0 : 1; ntr = 0                          # 히스테리시스: 아래 x<35, 위 x>100
    for v in rec
        if state == 0 && v > 100
            state = 1; ntr += 1
        elseif state == 1 && v < 35
            state = 0; ntr += 1
        end
    end
    hi = rec .> roots[2]
    @printf("  SSA from x0=62, t_max=%.0f, seed=42: events=%d, wall=%.2f s\n", tmax, nev, el)
    @printf("  fraction of time above unstable point=%.3f, basin-to-basin transitions=%d\n", mean(hi), ntr)
    @printf("  mean dwell time per visit=%.1f (time units)\n", tmax / (ntr + 1))
    @printf("  mean on low branch=%.1f, mean on high branch=%.1f\n", mean(rec[.!hi]), mean(rec[hi]))
end

main()
