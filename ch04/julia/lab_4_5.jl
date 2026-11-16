# Lab 4.5 (Julia) — 최소 ML 퍼텐셜: LJ PES를 RBF 커널 능형회귀(KRR)로 학습해 MD에 투입
# 파이썬 판(lab_4_5.py)과 같은 하이퍼파라미터(gamma=2, lambda=1e-8), 같은 학습 구간 [0.92, 2.5],
# 같은 MD 설정(2D, N=64, rho=0.7, T0=0.7, dt=0.004, 4000 step, 20 step마다 속도 재조정).
# 학습 표본과 초기속도가 난수이므로 RMSE·g(r) 수치는 파이썬과 "통계적으로만" 일치한다.
using LinearAlgebra, Printf, Random, Statistics
Random.seed!(2)

U_true(r) = 4.0 * (r^-12 - r^-6)
F_true(r) = 48.0 * r^-13 - 24.0 * r^-7
make_data(n) = (r = 0.92 .+ (2.5 - 0.92) .* rand(n); (r, U_true.(r)))   # np.random.uniform(0.92, 2.5, n)

struct KRR
    r::Vector{Float64}       # 학습 거리
    alpha::Vector{Float64}   # 회귀 계수
    gamma::Float64           # RBF 폭
end
function fit_krr(r, y; gamma = 2.0, lam = 1e-8)
    K = @. exp(-gamma * (r - r')^2)
    return KRR(copy(r), (K + lam * I) \ y, gamma)       # 선형계 풀기: np.linalg.solve
end
U(m::KRR, rq::Real) = sum(m.alpha[k] * exp(-m.gamma * (rq - m.r[k])^2) for k in eachindex(m.r))
"F = -dU/dr : RBF 커널의 해석적 미분"
F(m::KRR, rq::Real) = -sum(m.alpha[k] * exp(-m.gamma * (rq - m.r[k])^2) * (-2 * m.gamma * (rq - m.r[k]))
                           for k in eachindex(m.r))
rmse(a, b) = sqrt(mean((a .- b) .^ 2))

rg = collect(range(0.95, 2.5; length = 400))
println("--- learning curve: U/force RMSE vs #training points ---")
println("  Ntrain   U-RMSE     F-RMSE")
for n in (10, 20, 40, 80, 160, 320)
    rt, ut = make_data(n)
    m = fit_krr(rt, ut)
    @printf("  %5d   %.3e  %.3e\n", n, rmse(U.(Ref(m), rg), U_true.(rg)), rmse(F.(Ref(m), rg), F_true.(rg)))
end
rt, ut = make_data(160); model = fit_krr(rt, ut)
@printf("\nfinal model (160 pts): U-RMSE=%.2e, F-RMSE=%.2e\n",
        rmse(U.(Ref(model), rg), U_true.(rg)), rmse(F.(Ref(model), rg), F_true.(rg)))

# ---------------- MD: 힘 함수만 교체 (Lab 4.4 엔진, 2 x N 배치) ----------------
const rc2 = 2.5^2
function init_lattice(nside, density)
    N = nside * nside; L = sqrt(N / density); a = L / nside
    xs = ((0:nside-1) .+ 0.5) .* a
    pos = zeros(2, N); k = 0
    for i in 1:nside, j in 1:nside
        k += 1; pos[1, k] = xs[j]; pos[2, k] = xs[i]
    end
    return pos, L, N
end
"쌍 힘. fr(r)은 스칼라 쌍힘 크기 함수(F_true 또는 학습된 F) — '힘 루틴 교체'가 이 인자 하나"
function md_forces(pos, L, fr)
    N = size(pos, 2); Fo = zeros(2, N)
    for i in 1:N-1, j in i+1:N
        dx = pos[1, i] - pos[1, j]; dy = pos[2, i] - pos[2, j]
        dx -= L * round(dx / L); dy -= L * round(dy / L)
        r2 = dx * dx + dy * dy
        if 0.81 < r2 < rc2
            rr = sqrt(r2); s = fr(rr) / rr
            Fo[1, i] += s * dx; Fo[2, i] += s * dy
            Fo[1, j] -= s * dx; Fo[2, j] -= s * dy
        end
    end
    return Fo
end
function run_md(fr; nsteps = 4000, dt = 0.004, nside = 8, density = 0.7, T0 = 0.7, collect_after = 1500)
    pos, L, N = init_lattice(nside, density)
    vel = randn(2, N); vel .-= mean(vel; dims = 2)
    KE = 0.5 * sum(abs2, vel); T = 2KE / (2N - 2); vel .*= sqrt(T0 / T)
    Fo = md_forces(pos, L, fr)
    nbins = 100; rmax = L / 2; gr = zeros(nbins); nsnap = 0
    edges = collect(range(0, rmax; length = nbins + 1))
    for s in 0:nsteps-1
        vel .+= 0.5 * dt .* Fo; pos = mod.(pos .+ dt .* vel, L)
        Fo = md_forces(pos, L, fr); vel .+= 0.5 * dt .* Fo
        KE = 0.5 * sum(abs2, vel); T = 2KE / (2N - 2)
        s % 20 == 0 && (vel .*= sqrt(T0 / T))
        if s >= collect_after && s % 10 == 0
            for i in 1:N-1, j in i+1:N
                dx = pos[1, i] - pos[1, j]; dy = pos[2, i] - pos[2, j]
                dx -= L * round(dx / L); dy -= L * round(dy / L)
                d = sqrt(dx * dx + dy * dy)
                if d < rmax                                 # np.histogram(range=(0,rmax)) 과 같은 등간격 칸
                    b = min(floor(Int, d / rmax * nbins) + 1, nbins)   # 1-based 칸 번호
                    gr[b] += 1
                end
            end
            nsnap += 1
        end
    end
    rmid = 0.5 .* (edges[2:end] .+ edges[1:end-1])
    ring = pi .* (edges[2:end] .^ 2 .- edges[1:end-1] .^ 2); rho = N / L^2
    g = (gr ./ nsnap) ./ (((N - 1) / 2) .* rho .* ring)
    return rmid, g
end
nanmean(v) = mean(filter(isfinite, v))
r_t, g_t = run_md(F_true)
r_m, g_m = run_md(r -> F(model, r))                     # ← 바뀐 것은 이 한 줄(힘의 출처)뿐
ip, ipm = argmax(g_t), argmax(g_m)
@printf("  true-LJ g(r) first peak: r=%.3f, g=%.3f\n", r_t[ip], g_t[ip])
@printf("  ML      g(r) first peak: r=%.3f, g=%.3f\n", r_m[ipm], g_m[ipm])
sel = r_t .> 0.9
@printf("  g(r) RMSE (r>0.9): %.3f\n", sqrt(nanmean((g_t[sel] .- g_m[sel]) .^ 2)))
@printf("  g(r) tail mean (r>3): true=%.3f, ml=%.3f\n", nanmean(g_t[r_t .> 3]), nanmean(g_m[r_m .> 3]))
