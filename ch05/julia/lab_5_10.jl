# Lab 5.10 (Julia) — 네 구현의 속도와 정확성
# 파이썬 labs/ising_speed.py 대응. 같은 2D Ising Metropolis 물리를 Julia 안에서 네(다섯) 방식으로 구현한다.
#   naive_any    : 파이썬 "순수 Python"에 해당하는 것. Julia는 인터프리터 단계가 없으므로, 느린 코드의
#                  전형인 **타입 불안정**(Matrix{Any}, 매번 exp)으로 흉내 낸다.
#   vec_cb       : NumPy 체커보드에 해당. circshift + 브로드캐스트, 반-sweep마다 임시 배열.
#   loop_random  : Numba 무작위 자리(Ch.1 알고리즘)에 해당. 타입이 안정된 평범한 루프.
#   loop_cb      : Numba 체커보드 + 수용확률 표 (e^{-4 beta}, e^{-8 beta}).
#   batch_vec_cb : JAX vmap에 해당하는 알고리즘 내용, 즉 여러 온도를 L x L x NT 배열 하나로 묶어
#                  같은 브로드캐스트 체커보드를 한꺼번에 돌린다. XLA 컴파일, Threefry 카운터 RNG,
#                  GPU 실행은 Julia에 없으므로 **시간은 JAX와 비교할 수 없다**(★Why 5.11).
# 절대 시간과 비율은 기계·부하·Julia 버전에 따라 다르다. 정확성 부분은 Onsager(에너지)·Yang(자화)
# 정확해와 비교하며, 난수가 numpy와 다르므로 오차막대 안에서 통계적으로 일치한다.
# Onsager 에너지의 완전 타원적분 K(m)은 산술-기하 평균(AGM)으로 계산한다(SpecialFunctions 불필요).
using Printf, Random, Statistics

const J = 1.0
const TC = 2.0 / log(1.0 + sqrt(2.0))

# ---------------------------------------------------------------- 정확해
"제1종 완전 타원적분 K(m), 매개변수 m = k^2 (scipy.special.ellipk와 같은 규약), AGM"
function ellipk(m)
    a, b = 1.0, sqrt(1.0 - m)
    while abs(a - b) > 1e-15 * a
        a, b = (a + b) / 2, sqrt(a * b)
    end
    return pi / (2a)
end
function onsager_energy(T)
    b = 1.0 / T; k = 2.0 * sinh(2b) / cosh(2b)^2
    return -coth(2b) * (1 + (2 / pi) * (2 * tanh(2b)^2 - 1) * ellipk(k * k))
end
yang_m(T) = T < TC ? (1 - sinh(2 / T)^-4)^0.125 : 0.0

# ---------------------------------------------------------------- 1) 타입 불안정 "naive"
function sweep_naive!(s::Matrix{Any}, L, beta)
    for _ in 1:L*L
        i = rand(1:L); j = rand(1:L)
        nb = s[mod1(i + 1, L), j] + s[mod1(i - 1, L), j] + s[i, mod1(j + 1, L)] + s[i, mod1(j - 1, L)]
        dE = 2 * J * s[i, j] * nb
        if dE <= 0 || rand() < exp(-beta * dE)
            s[i, j] = -s[i, j]
        end
    end
end

# ---------------------------------------------------------------- 2) 브로드캐스트 체커보드 (NumPy 대응)
make_masks(L) = [[(i + j) % 2 == c for i in 1:L, j in 1:L] for c in (0, 1)]
function sweep_vec!(s, beta, masks)
    for m in masks                                  # 같은 색 자리끼리는 이웃이 아니다
        nb = circshift(s, (1, 0)) .+ circshift(s, (-1, 0)) .+ circshift(s, (0, 1)) .+ circshift(s, (0, -1))
        acc = (rand(size(s)...) .< exp.(-beta .* (2J .* s .* nb))) .& m
        s[acc] .*= -1
    end
end

# ---------------------------------------------------------------- 3) 4) 타입 안정 루프 (Numba 대응)
function sweeps_loop_random!(s, beta, nsweep)       # Ch.1 알고리즘: 무작위 자리
    L = size(s, 1); w4 = exp(-4beta * J); w8 = exp(-8beta * J)
    @inbounds for _ in 1:nsweep*L*L
        i = rand(1:L); j = rand(1:L)
        x = s[i, j] * (s[mod1(i + 1, L), j] + s[mod1(i - 1, L), j] + s[i, mod1(j + 1, L)] + s[i, mod1(j - 1, L)])
        if x <= 0                                    # dE = 2 J x, x in {-4,-2,0,2,4}
            s[i, j] = -s[i, j]
        elseif x == 2
            rand() < w4 && (s[i, j] = -s[i, j])
        else
            rand() < w8 && (s[i, j] = -s[i, j])
        end
    end
end

function sweeps_loop_cb!(s, beta, nsweep)           # 체커보드 순서 + 표 조회
    L = size(s, 1); w4 = exp(-4beta * J); w8 = exp(-8beta * J)
    @inbounds for _ in 1:nsweep, c in 0:1, j in 1:L
        jp = j < L ? j + 1 : 1; jm = j > 1 ? j - 1 : L
        for i in (mod(j - 1 + c, 2) + 1):2:L         # column-major: 열 안에서 연속 접근
            ip = i < L ? i + 1 : 1; im = i > 1 ? i - 1 : L
            x = s[i, j] * (s[ip, j] + s[im, j] + s[i, jp] + s[i, jm])
            if x <= 0
                s[i, j] = -s[i, j]
            elseif x == 2
                rand() < w4 && (s[i, j] = -s[i, j])
            elseif rand() < w8
                s[i, j] = -s[i, j]
            end
        end
    end
end

function measure(s)
    L = size(s, 1); e = 0.0; m = 0.0
    @inbounds for j in 1:L, i in 1:L
        e -= s[i, j] * (s[mod1(i + 1, L), j] + s[i, mod1(j + 1, L)]); m += s[i, j]
    end
    return e / L^2, m / L^2
end

# ---------------------------------------------------------------- 5) 온도 배치 (JAX vmap의 알고리즘 내용)
"S: L x L x NT, betas: 길이 NT. 모든 온도를 하나의 브로드캐스트로 한 sweep 진행"
function sweep_batch!(S, betas, masks3)
    B = reshape(betas, 1, 1, :)
    for m in masks3
        nb = circshift(S, (1, 0, 0)) .+ circshift(S, (-1, 0, 0)) .+ circshift(S, (0, 1, 0)) .+ circshift(S, (0, -1, 0))
        acc = (rand(size(S)...) .< exp.(-B .* (2J .* S .* nb))) .& m
        S[acc] .*= -1
    end
end
function measure_batch(S)
    L = size(S, 1)
    e = -vec(sum(S .* (circshift(S, (1, 0, 0)) .+ circshift(S, (0, 1, 0))); dims = (1, 2))) ./ L^2
    return e, vec(abs.(sum(S; dims = (1, 2)))) ./ L^2
end

"20블록 평균의 (평균, 표준오차)"
function blocking_err(x; nb = 20)
    n = length(x) ÷ nb; b = [mean(@view x[(k-1)*n+1:k*n]) for k in 1:nb]
    return mean(b), std(b) / sqrt(nb)
end

function main()
    L = 64; N = L * L; beta = 1 / 2.4; NT = 8; Ts = collect(range(1.6, 3.0; length = NT))
    Random.seed!(1)
    s_any = Matrix{Any}(fill(1, L, L)); s_vec = ones(Int, L, L); s_lp = ones(Int, L, L)
    masks = make_masks(L); masks3 = [repeat(m, 1, 1, NT) for m in masks]
    S0 = ones(Int, L, L, NT)
    t0 = time(); sweeps_loop_random!(s_lp, beta, 1); sweeps_loop_cb!(s_lp, beta, 1); t_jit = time() - t0
    t0 = time(); sweep_batch!(S0, 1 ./ Ts, masks3); t_jit_b = time() - t0
    sweep_naive!(s_any, L, beta); sweep_vec!(s_vec, beta, masks)                 # 나머지도 예열
    jobs = [("naive_any", () -> foreach(_ -> sweep_naive!(s_any, L, beta), 1:10), 10),
            ("vec_cb", () -> foreach(_ -> sweep_vec!(s_vec, beta, masks), 1:300), 300),
            ("loop_random", () -> sweeps_loop_random!(s_lp, beta, 1000), 1000),
            ("loop_cb", () -> sweeps_loop_cb!(s_lp, beta, 1000), 1000),
            ("batch_vec_cb$NT", () -> foreach(_ -> sweep_batch!(S0, 1 ./ Ts, masks3), 1:200), 200 * NT)]
    res = Dict(k => Inf for (k, _, _) in jobs)
    for _ in 1:5, (k, f, reps) in jobs                                          # 번갈아 5회, 최솟값
        res[k] = min(res[k], (@elapsed f()) / reps / N * 1e9)
    end
    @printf("=== Timing: L=%d (%d spins), T=2.4, ns per attempted spin update (best of 5) ===\n", L, N)
    base = res["naive_any"]
    for (k, _, _) in jobs
        @printf("  %-14s: %8.1f ns/update   speed-up vs naive x%6.0f\n", k, res[k], base / res[k])
    end
    @printf("  first call incl. JIT compile: loops %.2f s, batch %.2f s\n", t_jit, t_jit_b)

    println("\n=== Correctness: L=64, 1e4 equil + 5e4 meas. sweeps (loop_cb), 20 blocks ===")
    for T in (1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0)
        Random.seed!(7); s = ones(Int, L, L); b = 1 / T
        sweeps_loop_cb!(s, b, 10000)
        E = zeros(50000); M = zeros(50000)
        for n in 1:50000
            sweeps_loop_cb!(s, b, 1); E[n], M[n] = measure(s)
        end
        e, de = blocking_err(E); m, dm = blocking_err(abs.(M))
        @printf("  T=%.2f: e=%+.4f+/-%.4f (exact %+.4f)   |m|=%.4f+/-%.4f (Yang %.4f)\n",
                T, e, de, onsager_energy(T), m, dm, yang_m(T))
    end

    @printf("\n=== batched (vmap-like) over %d temperatures: 2e3 equil + 1e4 meas. sweeps ===\n", NT)
    Random.seed!(2); S = ones(Int, L, L, NT); bs = 1 ./ Ts
    t0 = time()
    for _ in 1:2000
        sweep_batch!(S, bs, masks3)
    end
    Ej = zeros(10000, NT); Mj = zeros(10000, NT)
    for n in 1:10000
        sweep_batch!(S, bs, masks3); Ej[n, :], Mj[n, :] = measure_batch(S)
    end
    @printf("  wall time %.1f s for %d x 12000 sweeps\n", time() - t0, NT)
    for (i, T) in enumerate(Ts)
        e, de = blocking_err(Ej[:, i]); m, dm = blocking_err(Mj[:, i])
        @printf("  T=%.2f: e=%+.4f+/-%.4f (exact %+.4f)  |m|=%.3f+/-%.3f (Yang %.3f)\n",
                T, e, de, onsager_energy(T), m, dm, yang_m(T))
    end
end

main()
