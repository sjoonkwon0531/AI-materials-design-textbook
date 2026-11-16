# lab_8_16.jl — Lab 8.16: 작은 규모의 DRO 재현, 학습된 최적화기 대 Nelder–Mead·CMA-ES·무작위 탐색·GP-EI (labs/86_dro.py)
# 지형: 3차원 단위 입방체 위의 가우스 혼합(K = 6, 폭 c = 0.3), 값은 단일 모드 높이로 0–1 정규화.
# 정책: 2층 LSTM(은닉 64; 게이트 i, f, g, o를 손으로 쓴 셀) -> 평균 d개와 d x d 하삼각 행렬, x_{t+1} = 평균 + tril(M) ξ.
# 손실: 미분 가능한 모의 지형을 통과하는 역전파(learning to learn). 'oi'(관측 개선)와 'bsf'(지금까지의 최고) 두 가지.
# 기준선: Nelder–Mead는 Optim.jl, CMA-ES는 아래의 최소 구현(Evolutionary.jl의 CMAES로 바꿔도 된다), 무작위 탐색, 최소 GP-EI.
# 학습(600 + 1,200회, 배치 128, 50단계 펼침)이 이 Lab의 무거운 부분이다.
include("ch8_common.jl")
include(joinpath(CH7_JL, "flux_tools.jl"))            # tdense
using Flux, Optimisers, Zygote, Optim, SpecialFunctions, NPZ
import ChainRulesCore

const D, K_TR, C_TR, HORIZON = 3, 6, 0.3, 50

# ---------------------------------------------------------------- 지형
"표준 정규를 [-2, 2]로 자른 표본 (거절 표집)"
function truncn(rng, dims...)
    z = randn(rng, dims...)
    while any(abs.(z) .> 2)
        z = ifelse.(abs.(z) .> 2, randn(rng, dims...), z)
    end
    return z
end
"m, s: d x K x n, w: K x n; top/bot: 단일 모드 높이 w/Πs의 최대·최소"
function sample_gmm(n, rng; K = K_TR, c = C_TR)
    m = 0.01 .+ 0.98 .* rand(rng, D, K, n)
    s = c .+ (c / 5) .* truncn(rng, D, K, n)
    w = 0.2 .* randn(rng, K, n)
    modes = w ./ dropdims(prod(s; dims = 1); dims = 1)
    return (m = m, s = s, w = w, top = vec(maximum(modes; dims = 1)), bot = vec(minimum(modes; dims = 1)))
end
"지형마다 한 점: X (d x n) -> 값 (n). Zygote로 미분 가능"
function fland(G, X)
    z = dropdims(sum(((reshape(X, D, 1, :) .- G.m) ./ G.s) .^ 2; dims = 1); dims = 1)              # K x n
    val = vec(sum(G.w ./ dropdims(prod(G.s; dims = 1); dims = 1) .* exp.(-0.5 .* z); dims = 1))
    return (val .- G.bot) ./ (G.top .- G.bot)
end
"지형 i 하나, 점 x 하나"
fone(G, i, x) = fland((m = G.m[:, :, i:i], s = G.s[:, :, i:i], w = G.w[:, i:i], top = G.top[i:i], bot = G.bot[i:i]), reshape(x, D, 1))[1]
"전역 최대: 31³ 격자, 그다음 가장 좋은 격자점 5개에서 상자 제약 L-BFGS로 다듬는다"
function global_max(G)
    g = range(0, 1; length = 31); grid = reduce(hcat, [[a, b, c] for a in g for b in g for c in g])
    map(1:length(G.top)) do i
        Gi = (m = G.m[:, :, i:i], s = G.s[:, :, i:i], w = G.w[:, i:i], top = G.top[i:i], bot = G.bot[i:i])
        v = fland(Gi, grid)   # fland는 점마다 지형 열을 짝지으므로, 지형 하나를 격자 전체에 펼친다
        best = maximum(v)
        for j in partialsortperm(v, 1:5; rev = true)
            r = optimize(x -> -fone(G, i, x), zeros(D), ones(D), clamp.(grid[:, j], 1e-6, 1 - 1e-6), Fminbox(LBFGS()))
            best = max(best, -Optim.minimum(r))
        end
        best
    end
end
x0_batch(n, rng) = clamp.(0.5 .+ 0.2 .* randn(rng, D, n), 0.1, 0.9)

# ---------------------------------------------------------------- DRO 정책: 2층 LSTM
struct LSTMCellJ{A,B}
    Wi::A            # 입력 -> 4h (편향 포함; torch의 b_ih + b_hh를 하나로)
    Wh::B            # 은닉 -> 4h (편향 없음)
end
Flux.@layer LSTMCellJ
function (c::LSTMCellJ)(x, h, cs)
    z = c.Wi(x) .+ c.Wh(h); H = size(h, 1)
    i, f, g, o = sigmoid.(z[1:H, :]), sigmoid.(z[H+1:2H, :]), tanh.(z[2H+1:3H, :]), sigmoid.(z[3H+1:4H, :])
    cs = f .* cs .+ i .* g
    return o .* tanh.(cs), cs
end
struct DRO{A,B,C}
    l1::A
    l2::B
    proj::C
end
Flux.@layer DRO
function DRO(rng; h = 64)
    LC(nin) = LSTMCellJ(tdense(rng, nin, 4h), tdense(rng, h, 4h; bias = false))   # torch nn.LSTM 초기화 U(±1/√h)와 다름
    return DRO(LC(2D), LC(h), tdense(rng, h, D + D * D))
end
const TRIL = Float32[i >= j for i in 1:D, j in 1:D]
"한 단계: (x, 관측값 y) -> (평균, 하삼각 L, 상태)"
function dro_step(net, x, y, st)
    inp = vcat(x, repeat(reshape(y, 1, :), D))
    h1, c1 = net.l1(inp, st[1], st[2]); h2, c2 = net.l2(h1, st[3], st[4])
    o = net.proj(h2)
    return o[1:D, :], reshape(o[D+1:end, :], D, D, :) .* TRIL, (h1, c1, h2, c2)
end
"직통(straight-through) 자르기: 값은 [0, 1] 안, 기울기는 자르지 않은 것처럼"
st_clamp(x) = x .+ ChainRulesCore.ignore_derivatives(clamp.(x, 0.0f0, 1.0f0) .- x)
"학습 손실을 펼침 안에서 바로 누적한다 (값을 배열에 쌓지 않으므로 Zygote에 편하다)"
function dro_loss(net, G, x0, rng; loss_type = :bsf, steps = HORIZON, hdim = 64)
    B = size(x0, 2); x = Float32.(x0)
    st = ntuple(_ -> zeros(Float32, hdim, B), 4)
    y = Float32.(fland(G, x)); best = y; acc = zero(y)
    for t in 1:steps
        mu, L, st = dro_step(net, x, y, st)
        xi = ChainRulesCore.ignore_derivatives(randn(rng, Float32, D, B))
        x = st_clamp(mu .+ dropdims(sum(L .* reshape(xi, 1, D, B); dims = 2); dims = 2))
        y = Float32.(fland(G, x))
        acc = loss_type == :oi ? acc .+ (y .- best) : acc     # oi: Σ_t [f(x_t) - max_{s<t} f(x_s)]
        best = max.(best, y)
        acc = loss_type == :oi ? acc : acc .+ best            # bsf: Σ_t max_{s≤t} f(x_s)
    end
    return -mean(acc)
end
"평가용 펼침 (기울기 없음). 반환: 참 값 (n x 51). blind면 정책에 관측 대신 0.5를 준다(Lab 8.16의 탐침)"
function run_dro(net, G, x0; stochastic = true, rng = Xoshiro(0), noise = 0.0, nrng = Xoshiro(0), blind = false, hdim = 64)
    B = size(x0, 2); x = Float32.(x0); st = ntuple(_ -> zeros(Float32, hdim, B), 4); Y = zeros(B, HORIZON + 1)
    for t in 0:HORIZON
        y = fland(G, x); Y[:, t+1] = y
        t == HORIZON && break
        yo = blind ? fill(0.5, B) : (noise == 0 ? y : y .+ noise .* randn(nrng, B))
        mu, L, st = dro_step(net, x, Float32.(yo), st)
        x = stochastic ? mu .+ dropdims(sum(L .* reshape(randn(rng, Float32, D, B), 1, D, B); dims = 2); dims = 2) : mu
        x = clamp.(x, 0.0f0, 1.0f0)
    end
    return Y
end
function train_dro(; iters = 1500, batch = 128, seed = 0, loss_type = :oi, lr = 1e-3, log_every = 250)
    rng = Xoshiro(seed); θ, re = Flux.destructure(DRO(rng)); opt = Optimisers.setup(Optimisers.Adam(lr), θ)
    for it in 1:iters
        G = sample_gmm(batch, rng); x0 = x0_batch(batch, rng)
        L, gs = Zygote.withgradient(q -> dro_loss(re(q), G, x0, rng; loss_type), θ)
        opt, θ = Optimisers.update!(opt, θ, clamp.(gs[1], -0.1f0, 0.1f0))      # 원소별 ±0.1 자르기 (저자 코드)
        it % log_every == 0 && @printf("    iter %5d  loss %8.3f\n", it, L)
    end
    return re(θ)
end

# ---------------------------------------------------------------- 기준선 (지형 하나씩, 예산 51회)
pad(v; budget = HORIZON + 1) = length(v) >= budget ? v[1:budget] : [v; fill(isempty(v) ? 0.0 : v[end], budget - length(v))]
function nelder_mead(fun, x0)
    seen = Float64[]
    f = x -> (v = fun(clamp.(x, 0, 1)); push!(seen, v); -v)
    optimize(f, x0, NelderMead(), Optim.Options(f_calls_limit = HORIZON + 1, iterations = 10_000, g_tol = 1e-12))
    return seen
end
"최소 (μ/μ_w, λ)-CMA-ES (Hansen의 기본 매개변수). 상자는 평가 점을 자르는 것으로만 다룬다"
function cmaes(fun, x0, seed; sigma = 0.2, budget = HORIZON + 1)
    rng = Xoshiro(seed + 1); n = length(x0); lam = 4 + floor(Int, 3log(n)); mu = lam ÷ 2
    w = log(mu + 0.5) .- log.(1:mu); w ./= sum(w); mueff = 1 / sum(abs2, w)
    cc = (4 + mueff / n) / (n + 4 + 2mueff / n); cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3)^2 + mueff); cmu = min(1 - c1, 2(mueff - 2 + 1 / mueff) / ((n + 2)^2 + mueff))
    damps = 1 + 2max(0, sqrt((mueff - 1) / (n + 1)) - 1) + cs; chiN = sqrt(n) * (1 - 1 / (4n) + 1 / (21n^2))
    m = copy(x0); pc, ps, C = zeros(n), zeros(n), Matrix(1.0I, n, n)
    seen = [fun(x0)]                                                           # 모든 방법이 같은 첫 실험
    while length(seen) < budget
        E = eigen(Symmetric(C)); Bm, Dm = E.vectors, sqrt.(max.(E.values, 1e-20))
        Zs = randn(rng, n, lam); Ys = Bm * (Dm .* Zs); Xs = m .+ sigma .* Ys
        F = [(v = fun(clamp.(Xs[:, k], 0, 1)); push!(seen, v); -v) for k in 1:lam]
        o = sortperm(F)[1:mu]; yw = Ys[:, o] * w
        m = m .+ sigma .* yw
        ps = (1 - cs) .* ps .+ sqrt(cs * (2 - cs) * mueff) .* (Bm * ((Bm' * yw) ./ Dm))
        hs = norm(ps) / sqrt(1 - (1 - cs)^(2length(seen) / lam)) < (1.4 + 2 / (n + 1)) * chiN
        pc = (1 - cc) .* pc .+ hs * sqrt(cc * (2 - cc) * mueff) .* yw
        C = (1 - c1 - cmu) .* C .+ c1 .* (pc * pc' .+ (1 - hs) * cc * (2 - cc) .* C) .+ cmu .* (Ys[:, o] * Diagonal(w) * Ys[:, o]')
        sigma *= exp((cs / damps) * (norm(ps) / chiN - 1))
    end
    return seen
end
random_search(fun, x0, rng) = [fun(x0); [fun(rand(rng, D)) for _ in 1:HORIZON]]
"최소 GP-EI: RBF 커널(길이 0.2 고정, 표준화한 y에 단위 사전 분산), 매번 무작위 후보 2,000개"
function gp_ei(fun, x0, rng; n_cand = 2000, ls = 0.2, noise = 1e-4)
    Xs = [copy(x0)]; Y = [fun(x0)]
    k(A, B) = exp.(-0.5 .* (sum(abs2, A; dims = 1)' .+ sum(abs2, B; dims = 1) .- 2 .* (A' * B)) ./ ls^2)
    for t in 1:HORIZON
        Xa = reduce(hcat, Xs); mu0, sd0 = mean(Y), std(Y; corrected = false) + 1e-9; Yn = (Y .- mu0) ./ sd0
        Lc = cholesky(Symmetric(k(Xa, Xa) + noise * I)).L
        al = Lc' \ (Lc \ Yn)
        Cn = rand(rng, D, n_cand); Ks = k(Cn, Xa)
        mu = Ks * al; v = Lc \ Ks'; sd = sqrt.(clamp.(1 .- vec(sum(abs2, v; dims = 1)), 1e-12, Inf))
        z = (mu .- maximum(Yn)) ./ sd
        ei = (mu .- maximum(Yn)) .* (0.5 .* erfc.(-z ./ sqrt(2))) .+ sd .* exp.(-z .^ 2 ./ 2) ./ sqrt(2pi)
        xn = Cn[:, argmax(ei)]; push!(Xs, xn); push!(Y, fun(xn))
    end
    return Y
end

# ---------------------------------------------------------------- 평가
function evaluate(nets, G, Fstar, x0, label; noise = 0.0, seed = 0)
    n = length(Fstar); out = Dict{String,Matrix{Float64}}()
    for (nm, net) in nets
        out["$nm (stochastic)"] = run_dro(net, G, x0; rng = Xoshiro(seed), noise, nrng = Xoshiro(seed))
        nm == "DRO-bsf" && (out["$nm (deterministic)"] = run_dro(net, G, x0; stochastic = false, noise, nrng = Xoshiro(seed)))
    end
    for name in ("Nelder-Mead", "CMA-ES", "Random search", "GP-EI")
        rows = map(1:n) do i
            nrng = Xoshiro(1000 + i - 1); seen = Float64[]
            fun = x -> (v = fone(G, i, x); push!(seen, v); v + (noise > 0 ? noise * randn(nrng) : 0.0))   # 최적화기는 잡음 섞인 값을 본다
            name == "Nelder-Mead" ? nelder_mead(fun, x0[:, i]) : name == "CMA-ES" ? cmaes(fun, x0[:, i], i - 1) :
            name == "Random search" ? random_search(fun, x0[:, i], Xoshiro(2000 + i - 1)) : gp_ei(fun, x0[:, i], Xoshiro(3000 + i - 1))
            pad(seen)'                                                                  # 참 값의 기록
        end
        out[name] = reduce(vcat, rows)
    end
    println("\n[$label]  $n test landscapes; regret = f* - best TRUE value found so far (mean +/- s.e.)")
    @printf("  %-24s %14s %14s %14s %19s\n", "method", "after 10", "after 20", "after 50", "P(regret<0.05 @50)")
    for (k, Y) in sort(collect(out); by = first)
        reg = Fstar .- reduce(hcat, [cummax(Y[i, :]) for i in 1:n])'
        s(c) = @sprintf("%7.3f+/-%.3f", mean(reg[:, c]), std(reg[:, c]) / sqrt(n))
        @printf("  %-24s %s %s %s %19.2f\n", k, s(11), s(21), s(51), mean(reg[:, 51] .< 0.05))
    end
end

println("training DRO on fresh simulated landscapes (K = 6, c = 0.3, d = 3), batch 128")
println("  (1) loss 'oi', 600 iterations"); net_oi = train_dro(; iters = 600, loss_type = :oi, log_every = 100)
println("  (2) loss 'bsf', 1,200 iterations"); net_bsf = train_dro(; iters = 1200, loss_type = :bsf, log_every = 200)
nets = ["DRO-oi" => net_oi, "DRO-bsf" => net_bsf]
const TESTFILE = joinpath(CH8_DATA, "86_test_sets.npz")                           # 파이썬과 같은 시험 지형(선택)
for (tag, label, K, c, noise) in [("id", "in-distribution: K=6, c=0.3", 6, 0.3, 0.0), ("rugged", "OOD-rugged: K=12, c=0.15", 12, 0.15, 0.0),
                                  ("noisy", "OOD-noisy: K=6, c=0.3, sigma_y=0.05", 6, 0.3, 0.05)]
    if isfile(TESTFILE)
        z = npzread(TESTFILE)
        G = (m = permutedims(z["$(tag)_m"], (3, 2, 1)), s = permutedims(z["$(tag)_s"], (3, 2, 1)), w = permutedims(z["$(tag)_w"]),
             top = z["$(tag)_top"], bot = z["$(tag)_bot"])
        Fs, x0 = z["$(tag)_Fs"], permutedims(z["$(tag)_x0"])
    else
        rng = Xoshiro(99); G = sample_gmm(150, rng; K, c); Fs = global_max(G); x0 = x0_batch(150, rng)
    end
    evaluate(nets, G, Fs, x0, label; noise)
end
