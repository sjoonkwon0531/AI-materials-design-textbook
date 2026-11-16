# lab_9_5.jl — Lab 9.5: 비선형 문제 둘, Burgers 방정식의 충격과 1차원 Cahn–Hilliard의 혼합 정식화 (labs/92_pdes.py)
# (A) u_t + u u_x = ν u_xx, x ∈ [-1, 1], u(x, 0) = -sin πx, u(±1, t) = 0, ν = 0.01/π.
#     기준 해: Cole–Hopf 공식 + Gauss–Hermite 구적 120점(FastGaussQuadrature.jl). 기준선: 보존형 중심 차분 + OrdinaryDiffEq.jl.
#     PINN 2-32x4-1 tanh, 강제 구속 u = -sin πx + t (1 - x²) N(x, t).
# (B) c_t = (c³ - c - ε² c_xx)_xx, 주기 [0, 2π), ε = 0.25. 화학 퍼텐셜 μ를 두 번째 출력으로 둔 혼합 정식화(2계 도함수만 필요).
#     기준 해: Ch.2 Derivation Box 2.22의 안정화 반암시 스펙트럴 스킴(N = 256, dt = 2e-5).
# 줄인 포트: 시드 하나씩만 돌리고, 혼합 대 4계 직접 잔차의 비용 비교는 뺐다(jet을 4계까지 쓰면 된다; lab_9_4.jl의 jet4).
include("ch9_common.jl")
include("pinn_common.jl")
using OrdinaryDiffEq, FastGaussQuadrature, FFTW

# ================================================================ A. Burgers
const NU = 0.01 / pi
const ZQ, WQ = gausshermite(120)
"Cole–Hopf 해 (u0 = -sin πx, t > 0)"
function burgers_exact(x::AbstractVector, t)
    y = x .- sqrt(4 * NU * t) .* ZQ'                                   # 점 x 구적점
    f = exp.(-cos.(pi .* y) ./ (2pi * NU))
    return -((sin.(pi .* y) .* f) * WQ) ./ (f * WQ)
end
burgers_grid(x, ts) = reduce(hcat, [t == 0 ? -sin.(pi .* x) : burgers_exact(x, t) for t in ts])

"보존형 중심 차분의 선 방법. 이 점성에서는 강성이 아니어서 양해법(Tsit5)으로 충분하다"
function fd_burgers(N, ts)
    x = collect(range(-1, 1; length = N + 1)); dx = 2 / N
    function rhs!(du, u, _, t)
        n = length(u)
        for i in 1:n
            um = i == 1 ? 0.0 : u[i-1]; up = i == n ? 0.0 : u[i+1]
            du[i] = -(up^2 - um^2) / (4dx) + NU * (up - 2u[i] + um) / dx^2
        end
        return nothing
    end
    t0 = time()
    sol = solve(ODEProblem(rhs!, -sin.(pi .* x[2:end-1]), (0.0, 1.0)), Tsit5(); reltol = 1e-8, abstol = 1e-10, saveat = ts)
    w = time() - t0
    U = vcat(zeros(1, length(ts)), reduce(hcat, sol.u), zeros(1, length(ts)))
    return rel_l2(U, burgers_grid(x, ts)), w
end

const SB = [2, 32, 32, 32, 32, 1]
const EX, ET, Z2 = [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]
"잔차 u_t + u u_x - ν u_xx. g = t (1 - x²), g_x = -2tx, g_xx = -2t, g_t = 1 - x²"
function burgers_residual(θ, x, t)
    X = vcat(x, t)
    n, nx, nxx = jet2(SB, θ, X, EX, Z2); _, nt = jet1(SB, θ, X, ET)
    g = t .* (1 .- x .^ 2)
    u = -sin.(pi .* x) .+ g .* n
    ux = -pi .* cos.(pi .* x) .- 2 .* t .* x .* n .+ g .* nx
    uxx = pi^2 .* sin.(pi .* x) .- 2 .* t .* n .- 4 .* t .* x .* nx .+ g .* nxx
    ut = (1 .- x .^ 2) .* n .+ g .* nt
    return ut .+ u .* ux .- NU .* uxx
end
burgers_u(θ, x, t) = -sin.(pi .* x) .+ t .* (1 .- x .^ 2) .* mlp_value(SB, θ, vcat(x, t))

function partA(; seed = 0, n_adam = 1500, n_lbfgs = 2000, NF = 2000)
    xg = collect(range(-1, 1; length = 401)); tg = collect(range(0, 1; length = 101))
    UG = burgers_grid(xg, tg)                                          # 401 x 101
    xfine = collect(range(-0.02, 0.02; length = 4001)); tt = range(0.4, 0.6; length = 201)
    cdiff(u) = (u[3:end] .- u[1:end-2]) ./ (xfine[3] - xfine[1])           # 중심 차분 (np.gradient의 내부 점과 같다)
    sl = [maximum(abs, cdiff(burgers_exact(xfine, t))) for t in tt]
    @printf("=== A. Burgers, nu = 0.01/pi; reference = Cole-Hopf (120-point Gauss-Hermite); max |u_x| = %.2f at t = %.3f ===\n", maximum(sl), tt[argmax(sl)])
    for N in (100, 200, 400, 800, 1600)
        e, w = fd_burgers(N, tg)
        @printf("  FD N = %4d: rel L2 error %.2e, %6.0f ms\n", N, e, 1e3 * w)
    end
    _, _, θ = init_mlp(SB; seed); g = Xoshiro(300 + seed)
    xf = hcat(2 .* rand(g, 1, NF ÷ 2) .- 1, 0.2 .* (2 .* rand(g, 1, NF ÷ 2) .- 1))   # 절반은 균등하게, 절반은 앞면이 서는 x = 0 근처에
    tf = rand(g, 1, NF)
    loss(θ) = mean(abs2, burgers_residual(θ, xf, tf))
    XG = reshape([x for x in xg, t in tg], 1, :); TG = reshape([t for x in xg, t in tg], 1, :)
    err(θ) = rel_l2(reshape(burgers_u(θ, XG, TG), size(UG)), UG)
    t0 = time()
    θ = train_adam(loss, θ; iters = n_adam, lr = 2e-3); ea = err(θ)
    θ = train_lbfgs(loss, θ; iters = n_lbfgs)
    @printf("  PINN (2-32x4-1 tanh, hard IC/BC, %d points) seed %d: rel L2 after Adam %.2e -> after L-BFGS %.2e, final loss %.1e, %.0f s\n",
            NF, seed, ea, err(θ), loss(θ), time() - t0)
end

# ================================================================ B. 1차원 Cahn–Hilliard
const EPS, TCH = 0.25, 1.0
c0f(x) = 0.2 * cos(2x) + 0.1 * cos(3x)
c0xx(x) = -0.8 * cos(2x) - 0.9 * cos(3x)

"안정화 반암시 스펙트럴 스킴 (Box 2.22). C는 N x nout (열 = 시각)"
function ch_spectral(; N = 256, dt = 1e-4, S = 2.0, nout = 101)
    x = collect(0:N-1) .* (2pi / N); k = collect(fftfreq(N, N))
    k2 = k .^ 2; den = 1 .+ dt * S .* k2 .+ dt * EPS^2 .* k2 .^ 2; gain = 1 .+ dt * S .* k2
    c = c0f.(x); ch = fft(c); steps = round(Int, TCH / dt); every = steps ÷ (nout - 1); C = [copy(c)]
    for n in 1:steps
        ch = (gain .* ch .- dt .* k2 .* fft(c .^ 3 .- c)) ./ den; c = real.(ifft(ch))
        n % every == 0 && push!(C, copy(c))
    end
    return x, reduce(hcat, C)
end
"자유에너지 F(t) = ∫ (c² - 1)²/4 + ε² c_x²/2 dx (스펙트럴 도함수)"
function free_energy(C)
    k = fftfreq(size(C, 1), size(C, 1))
    cx = real.(ifft(im .* k .* fft(C, 1), 1))
    return vec(mean(0.25 .* (C .^ 2 .- 1) .^ 2 .+ 0.5 * EPS^2 .* cx .^ 2; dims = 1)) .* 2pi
end

const SC = [7, 48, 48, 48, 48, 2]
const ET7 = vcat(zeros(6), 1 / TCH)                                    # t 방향: 입력의 마지막 성분 t/T
"주기 임베딩 z(x, t)와 x에 대한 1·2계 도함수 (강제 주기 경계조건)"
function embed(x, t)
    z = vcat(cos.(x), sin.(x), cos.(2 .* x), sin.(2 .* x), cos.(3 .* x), sin.(3 .* x), t ./ TCH)
    zx = vcat(-sin.(x), cos.(x), -2 .* sin.(2 .* x), 2 .* cos.(2 .* x), -3 .* sin.(3 .* x), 3 .* cos.(3 .* x), zero(x))
    zxx = vcat(-cos.(x), -sin.(x), -4 .* cos.(2 .* x), -4 .* sin.(2 .* x), -9 .* cos.(3 .* x), -9 .* sin.(3 .* x), zero(x))
    return z, zx, zxx
end
"혼합 정식화의 두 잔차: r1 = c_t - μ_xx, r2 = μ - (c³ - c - ε² c_xx). c = c0(x) + (t/T) o₁ (강제 초기조건), μ = o₂"
function ch_residuals(θ, x, t, emb)
    z, zx, zxx = emb
    o, _, oxx = jet2(SC, θ, z, zx, zxx); _, ot = jet1(SC, θ, z, ET7)
    tau = t ./ TCH
    c = c0f.(x) .+ tau .* o[1:1, :]
    ct = o[1:1, :] ./ TCH .+ tau .* ot[1:1, :]
    cxx = c0xx.(x) .+ tau .* oxx[1:1, :]
    mu = o[2:2, :]
    return ct .- oxx[2:2, :], mu .- (c .^ 3 .- c .- EPS^2 .* cxx)
end
ch_c(θ, x, t) = c0f.(x) .+ (t ./ TCH) .* mlp_value(SC, θ, embed(x, t)[1])[1:1, :]

function partB(; seed = 0, n_adam = 1500, n_lbfgs = 2500, NF = 2000)
    t0 = time(); xs, CS = ch_spectral(; dt = 2e-5); w_ref = time() - t0
    _, CS2 = ch_spectral(; dt = 1e-5); Fs = free_energy(CS)
    @printf("=== B. 1D Cahn-Hilliard, eps = %.2f, T = %.1f; spectral reference N=256, dt=2e-5: %.0f ms ===\n", EPS, TCH, 1e3 * w_ref)
    @printf("  reference self-check: rel L2 diff between dt=2e-5 and dt=1e-5: %.1e\n", rel_l2(CS, CS2))
    @printf("  reference: max|c| %.2f -> %.2f; mass drift %.1e; F %.3f -> %.3f, increases %d/100\n", maximum(abs, CS[:, 1]), maximum(abs, CS[:, end]),
            maximum(abs, vec(mean(CS; dims = 1)) .- mean(CS[:, 1])), Fs[1], Fs[end], count(>(1e-12), diff(Fs)))
    tg = range(0, TCH; length = 101)
    XS = reshape([x for x in xs, t in tg], 1, :); TS = reshape([t for x in xs, t in tg], 1, :)
    _, _, θ = init_mlp(SC; seed); g = Xoshiro(400 + seed)
    xf = 2pi .* rand(g, 1, NF); tf = TCH .* rand(g, 1, NF); emb = embed(xf, tf)        # 임베딩은 상수: 한 번만 만든다
    function loss(θ)
        r1, r2 = ch_residuals(θ, xf, tf, emb)
        return mean(abs2, r1) + 10 * mean(abs2, r2)
    end
    pred(θ) = reshape(ch_c(θ, XS, TS), size(CS))
    t0 = time()
    θ = train_adam(loss, θ; iters = n_adam, lr = 2e-3); ea = rel_l2(pred(θ), CS)
    θ = train_lbfgs(loss, θ; iters = n_lbfgs)
    C = pred(θ); F = free_energy(C); r1, r2 = ch_residuals(θ, xf, tf, emb)
    relt(j) = norm(C[:, j] .- CS[:, j]) / norm(CS[:, j])
    @printf("  PINN mixed (c, mu) seed %d: rel L2 after Adam %.2e -> after L-BFGS %.2e; err(t=0.25,0.5,1) = %.2e, %.2e, %.2e\n", seed, ea, rel_l2(C, CS), relt(26), relt(51), relt(101))
    @printf("       residual losses r1 %.1e r2 %.1e; mass drift max|<c>(t)-<c>(0)| %.1e; F %.3f -> %.3f, increases %d/100; %.0f s\n",
            mean(abs2, r1), mean(abs2, r2), maximum(abs, vec(mean(C; dims = 1)) .- mean(C[:, 1])), F[1], F[end], count(>(1e-12), diff(F)), time() - t0)
end

partA(); partB()
