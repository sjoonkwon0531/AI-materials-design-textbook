# bandgap83.jl — Lab 8.9와 8.11의 공용 부분: 7장의 149개 특징 표, 파이썬과 같은 분할, 밴드갭 MLP, 학습, 날카로움
# 분할은 파이썬이 저장한 labs/83_bandgap_split.npz(0-based)를 읽는다. 검증 집합은 파이썬처럼 tr_all의 앞 1/8이다.
include("ch8_common.jl")
include(joinpath(CH7_JL, "ch7_common.jl"))          # load_features149, TorchAdamW, cosine_lr
include(joinpath(CH7_JL, "flux_tools.jl"))          # tdense, mlp_chain, batches, gap_mae
include("sam.jl")
using NPZ, Serialization, Flux

const XF, YF = load_features149()[1:2]              # 4,601 x 149, 밴드갭 (eV)
const YG = Float32.(YF)
const SPLIT = npzread(joinpath(CH8_LABS, "83_bandgap_split.npz"))
const TR_ALL, TE, OX, HA, SUB = (SPLIT[k] .+ 1 for k in ("tr_all", "te", "ox", "ha", "sub"))
const VA, TR = TR_ALL[1:length(TR_ALL)÷8], TR_ALL[length(TR_ALL)÷8+1:end]
const MU_X = mean(XF[TR_ALL, :]; dims = 1)
const SD_X = let s = std(XF[TR_ALL, :]; dims = 1, corrected = false)      # StandardScaler: 분산 0인 열은 1
    s[s .== 0] .= 1; s
end
"표준화 + ±6 자르기, 특징 x 관측 (Float32)"
Zf(idx) = Float32.(permutedims(clamp.((XF[idx, :] .- MU_X) ./ SD_X, -6, 6)))
const MODELDIR = joinpath(@__DIR__, "models"); mkpath(MODELDIR)

make_mlp(p; seed = 0) = mlp_chain(Xoshiro(seed), [p, 256, 256, 1], relu)
mae_on(model, idx) = gap_mae(model, Zf(idx), YG[idx])

"""AdamW(PyTorch식 분리 감쇠) + cosine 일정, 미니배치 64. method = :none(AdamW만), :sam, :asam, :fsam.
noise > 0이면 입력 잡음 증강, fgsm > 0이면 FGSM 적대 예제를 배치에 덧붙인다."""
function fit(Xt, yt, method; seed = 0, rho = 0.05, epochs = 60, bs = 64, lr = 1e-3, wd = 1e-4, noise = 0.0, fgsm = 0.0)
    θ, re = Flux.destructure(make_mlp(size(Xt, 1); seed))
    opt = Optimisers.setup(TorchAdamW(lr; lambda = wd), θ); ss = SAMState(); rng = Xoshiro(seed)
    n = length(yt); total = epochs * cld(n, bs); s = 0
    for ep in 1:epochs, j in batches(rng, n, bs)
        xb, yb = Xt[:, j], yt[j]
        noise > 0 && (xb = xb .+ Float32(noise) .* randn(rng, Float32, size(xb)))
        if fgsm > 0                                              # FGSM: 입력 기울기의 부호 방향으로 한 걸음
            m = re(θ); gx = only(Zygote.gradient(x -> mean(abs2, vec(m(x)) .- yb), xb))
            xb = hcat(xb, xb .+ Float32(fgsm) .* sign.(gx)); yb = vcat(yb, yb)
        end
        Optimisers.adjust!(opt, cosine_lr(lr, s, total)); s += 1
        opt, θ, _ = sam_update!(opt, θ, q -> mean(abs2, vec(re(q)(xb)) .- yb), ss; rho, mode = method)
    end
    return re(θ)
end

"""훈련 MSE의 λ_max (adaptive = true면 T H T, T = |w|). Float64 사본에서 정확한 HVP로 계산한다.
기본값(40회, 상대 변화 1e-4에서 멈춤)은 파이썬 판의 옛 규칙이다(파이썬 판은 이제 eigsh로 수렴시킨다). 고윳값 간격이 작으면 수렴 전에 멈춰 과소평가할 수 있다."""
function sharpness(model, idx; adaptive = false, iters = 40, tol = 1e-4)
    θ, L = chain_mse(model, Zf(idx), YG[idx])           # Float64 θ와 Dual을 받는 손실: 정확한 HVP(hvp_exact)용
    return adaptive ? lambda_max_adaptive(L, θ; iters, tol)[1] : lambda_max(L, θ; iters, tol)[1]
end

save_model(tag, seed, m) = serialize(joinpath(MODELDIR, "83_bg_$(tag)_s$(seed).jls"), m)
load_model(tag, seed) = deserialize(joinpath(MODELDIR, "83_bg_$(tag)_s$(seed).jls"))

"""파이썬이 저장한 83_bg_*.pt(state_dict: 0, 2, 4번 Linear)를 같은 구조의 Flux Chain(1, 3, 5번 Dense)으로 옮긴다.
부르기 전에 `using PythonCall`이 필요하다."""
function load_torch_mlp(tag, seed)
    torch = pyimport("torch"); sd = torch.load(joinpath(CH8_LABS, "83_bg_$(tag)_s$(seed).pt"))
    g(k) = pyconvert(Array{Float32}, sd[k].numpy())
    m = make_mlp(size(XF, 2))
    for (i, k) in zip((1, 3, 5), (0, 2, 4))
        m[i].weight .= g("$k.weight"); m[i].bias .= g("$k.bias")
    end
    return m
end
