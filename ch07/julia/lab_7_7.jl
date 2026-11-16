# lab_7_7.jl — Lab 7.7: 시간 역전파의 소실, 그리고 용량 열화 곡선 예측 (labs/72a_lstm.py)
# RNN과 LSTM 셀을 직접 썼다(게이트 순서 i, f, g, o는 PyTorch와 같다). Flux의 순환층 API가 0.14 -> 0.15/0.16에서
# 크게 바뀌었기 때문에, 판에 묶이지 않도록 셀 식을 그대로 적는 편을 택했다. 시퀀스 배치는 (입력, 시간, 배치).
# 용량 열화 곡선은 파이썬 판과 같은 합성 모형(√사이클 SEI + 선형 + 무릎 이후 2차)이다. 측정 자료가 아니다.
include("ch7_common.jl")
include("flux_tools.jl")

# ---------------------------------------------------------------- 셀
struct RNNh{M,V}
    Wi::M
    Wh::M
    b::V
end
Flux.@layer RNNh
struct LSTMh{M,V}
    Wi::M
    Wh::M
    b::V
end
Flux.@layer LSTMh
"PyTorch 기본 초기화 U(-1/√H, 1/√H); 편향은 b_ih + b_hh 두 벡터의 합. LSTM은 망각 게이트 편향을 fbias로 둔다"
function cell(kind, rng, nin, H; fbias = 0.0)
    u(d...) = Float32.((rand(rng, d...) .* 2 .- 1) ./ sqrt(H))
    if kind == :rnn
        return RNNh(u(H, nin), u(H, H), u(H) .+ u(H))
    end
    b = u(4H) .+ u(4H); b[H+1:2H] .= fbias
    return LSTMh(u(4H, nin), u(4H, H), b)
end
function (l::RNNh)(x)                      # x: (nin, T, B) -> h_T: (H, B)
    h = zeros(eltype(x), size(l.Wh, 1), size(x, 3))
    for t in 1:size(x, 2)
        h = tanh.(l.Wi * x[:, t, :] .+ l.Wh * h .+ l.b)
    end
    return h
end
function (l::LSTMh)(x)
    H = size(l.Wh, 2)
    h = zeros(eltype(x), H, size(x, 3)); c = h
    for t in 1:size(x, 2)
        z = l.Wi * x[:, t, :] .+ l.Wh * h .+ l.b
        i, f = sigmoid.(z[1:H, :]), sigmoid.(z[H+1:2H, :])
        g, o = tanh.(z[2H+1:3H, :]), sigmoid.(z[3H+1:4H, :])
        c = f .* c .+ i .* g
        h = o .* tanh.(c)
    end
    return h
end

# ---------------------------------------------------------------- Part A
function partA()
    println("=== Part A: ||d h_T / d x_t|| vs lag, hidden 32, input 1, T = 200, mean of 10 random inits ===")
    T, H = 200, 32; lags = [1, 10, 25, 50, 100, 199]
    for (kind, fb, name) in ((:rnn, 0.0, "vanilla RNN (tanh)"), (:lstm, 0.0, "LSTM, forget bias 0"), (:lstm, 3.0, "LSTM, forget bias +3"))
        G = zeros(T)
        for s in 0:9
            rng = Xoshiro(s); net = cell(kind, rng, 1, H; fbias = fb)
            x = randn(rng, Float32, 1, T, 1)
            g = only(Zygote.gradient(z -> sum(net(z)), x))
            G .+= reverse(abs.(vec(g))) ./ 10              # 인덱스 = 지연 + 1
        end
        println(rpad(name, 22), join([@sprintf("  lag %3d: %8.1e", l, G[l+1]) for l in lags]))
    end
end

# ---------------------------------------------------------------- Part B
"합성 열화 곡선: q = 1 - a√c - b c - d max(c - n_k, 0)² + 잡음, 용량 60%에서 멈춤"
function fade_cell(rng; n = 1200)
    a, b = 1e-3 + 2e-3 * rand(rng), 2e-5 + 4e-5 * rand(rng)
    nk, d = 400 + 500 * rand(rng), 1e-6 + 3e-6 * rand(rng)
    c = 0:n-1
    q = @. 1 - a * sqrt(c) - b * c - d * max(c - nk, 0)^2
    q .+= 0.002 .* randn(rng, n)
    stop = findfirst(<(0.6), q)
    return Float32.(q[1:(stop === nothing ? n : stop - 1)]), nk
end
const W, HZ = 50, 100
function windows(cs, stride)
    X, Y, K = Vector{Float32}[], Vector{Float32}[], Bool[]
    for (q, nk) in cs, s in 1:stride:length(q)-W-HZ
        push!(X, q[s:s+W-1]); push!(Y, q[s+W:s+W+HZ-1] .- q[s+W-1]); push!(K, s + W - 2 + HZ > nk)
    end
    return reduce(hcat, X), reduce(hcat, Y), K                      # (W, n), (HZ, n)
end

function partB()
    rng = Xoshiro(0)
    cells = [fade_cell(rng) for _ in 1:200]
    Xtr, Ytr, _ = windows(cells[1:150], 5); Xte, Yte, Kte = windows(cells[151:200], 10)
    @printf("=== Part B: windows train %d, test %d (horizon past the knee: %.2f) ===\n", size(Xtr, 2), size(Xte, 2), mean(Kte))
    function evaluate(pred)                                          # RMSE (% 공칭 용량): 전체, 무릎 전, 무릎, 지평별
        e = pred .- Yte
        rms(m) = sqrt(mean(abs2, e[:, m])) * 100
        return rms(trues(length(Kte))), rms(.!Kte), rms(Kte), vec(sqrt.(mean(abs2, e; dims = 2))) .* 100
    end
    res = Pair{String,Any}[]
    push!(res, "persistence" => evaluate(zero(Yte)))
    t = collect(0:W-1) .- (W - 1); hz = collect(1:HZ)
    slope = vec(sum((Xte .- mean(Xte; dims = 1)) .* (t .- mean(t)); dims = 1)) ./ sum(abs2, t .- mean(t))
    push!(res, "linear extrapolation" => evaluate(hz .* slope'))
    A = [ones(W) t t .^ 2]; coef = A \ Xte                           # 창마다 2차 맞춤 (3 x n)
    push!(res, "quadratic extrapolation" => evaluate(coef[2:2, :] .* hz .+ coef[3:3, :] .* hz .^ 2))
    prep(X) = (X .- X[end:end, :]) ./ 0.01f0                         # 마지막 값 기준, % 단위
    function fit(mk, seed; epochs = 25)
        r = Xoshiro(seed); net = mk(r)
        Xt, Yt = prep(Xtr), Ytr ./ 0.01f0
        opt = Flux.setup(TorchAdamW(2e-3; lambda = 1e-4), net)
        total = epochs * cld(size(Yt, 2), 256); step = 0
        for ep in 1:epochs, j in batches(r, size(Yt, 2), 256)
            Flux.adjust!(opt, cosine_lr(2e-3, step, total)); step += 1
            xb, yb = Xt[:, j], Yt[:, j]
            g = only(Flux.gradient(m -> mean(abs2, m(xb) .- yb), net))
            Flux.update!(opt, net, g)
        end
        return net(prep(Xte)) .* 0.01f0, net
    end
    mlp(r) = Chain(tdense(r, W, 128, swish), tdense(r, 128, 128, swish), tdense(r, 128, HZ))
    lstm(r) = Chain(x -> reshape(x, 1, size(x)...), cell(:lstm, r, 1, 64), tdense(r, 64, HZ))   # (W, B) -> (1, W, B)
    for (name, mk) in ("MLP" => mlp, "LSTM" => lstm)
        runs = []; net = nothing
        for seed in 0:2
            p, net = fit(mk, seed); push!(runs, evaluate(p))
        end
        push!(res, name => (mean(r[1] for r in runs), mean(r[2] for r in runs), mean(r[3] for r in runs), mean(r[4] for r in runs)))
        @printf("[%s: %d params]\n", name, nparams(net))
    end
    @printf("%-24s %9s %9s %9s %10s %8s   (%% of nominal capacity)\n", "model", "RMSE all", "pre-knee", "knee", "RMSE@h=10", "@h=100")
    for (k, r) in res
        @printf("%-24s %9.3f %9.3f %9.3f %10.3f %8.3f\n", k, r[1], r[2], r[3], r[4][10], r[4][end])
    end
end

partA(); partB()
