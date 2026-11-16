# lab_7_5.jl — Lab 7.5: 1D CNN으로 XRD 구조형 판별하기 (labs/72a_xrd_cnn.py)
# 회절 피크 목록(pymatgen XRDCalculator로 만든 72a_xrd_peaks.npz)은 객체 배열이라 NPZ.jl로 읽을 수 없어,
# 파이썬에서 한 번 CSV(data/xrd_peaks.csv: pid, two_theta, intensity / data/xrd_labels.csv: pid, label)로 내보낸다.
# 패턴 렌더링(가우스 피크, 무작위 FWHM, 2% 잡음), CNN·MLP, one-cycle 학습률, 영점 이동 시험을 옮겼다.
# Flux Conv에는 circular padding 모드가 없어 NNlib.pad_circular로 입력을 감싼 뒤 pad = 0으로 합성곱한다.
include("ch7_common.jl")
include("flux_tools.jl")

const SYS = ["fcc", "bcc", "hcp", "rocksalt", "CsCl", "zincblende", "wurtzite", "fluorite", "rutile", "perovskite"]
const TT = collect(10.0:0.1:79.95)                 # 700 채널
pk = CSV.read(joinpath(DATA, "xrd_peaks.csv"), DataFrame)
lab = CSV.read(joinpath(DATA, "xrd_labels.csv"), DataFrame)
PX = [pk.two_theta[pk.pid .== i] for i in lab.pid]
PY = [pk.intensity[pk.pid .== i] for i in lab.pid]
LAB = lab.label                                    # 0..9

"가우스 피크(FWHM 0.15–0.35°), 최대값 정규화, 2% 잡음; shift는 모든 2θ에 더하는 영점 이동"
function render(i, rng; shift = 0.0)
    w = (0.15 + 0.2 * rand(rng)) / 2.3548
    prof = vec(sum(PY[i] .* exp.(-0.5 .* ((TT' .- PX[i] .- shift) ./ w) .^ 2); dims = 1))
    return Float32.(prof ./ maximum(prof) .+ 0.02 .* randn(rng, length(TT)))
end

rng = Xoshiro(1)
idx = randperm(rng, length(LAB)); n_te = 1000
te_i, tr_i = idx[1:n_te], idx[n_te+1:end]
Xtr = reshape(reduce(hcat, [render(i, rng) for i in tr_i]), length(TT), 1, :)   # (700, 1, B)
ytr = LAB[tr_i]; yte = LAB[te_i]
test_set(shift; seed = 2) = (r = Xoshiro(seed); reshape(reduce(hcat, [render(i, r; shift) for i in te_i]), length(TT), 1, :))

conv1(rng, a, b, k) = Conv((k,), a => b, swish; pad = k ÷ 2, init = Flux.kaiming_uniform(rng; gain = sqrt(2 / 6)))
function CNN(rng; head = :gap, deep = false)
    ch = deep ? [1, 16, 32, 32, 64, 64] : [1, 16, 32, 32]
    feat = Chain([Chain(conv1(rng, ch[k], ch[k+1], 7), MaxPool((2,))) for k in 1:length(ch)-1]...)
    L = 700
    for _ in 1:length(ch)-1
        L ÷= 2
    end
    pool = head == :gap ? (h -> dropdims(maximum(h; dims = 1); dims = 1)) : Flux.flatten   # 전역 최대 풀링(불변) 또는 펼침(아님)
    return Chain(feat, pool, tdense(rng, head == :gap ? ch[end] : ch[end] * L, 10))
end
MLPx(rng) = Chain(Flux.flatten, tdense(rng, 700, 256, swish), tdense(rng, 256, 128, swish), tdense(rng, 128, 10))

function train(net, X, y; epochs = 20, seed = 0, aug_shift = 0.0)
    rng = Xoshiro(seed)
    opt = Flux.setup(TorchAdamW(2e-3; lambda = 1e-2), net)
    total = epochs * cld(length(y), 64); step = 0
    for ep in 1:epochs, j in batches(rng, length(y), 64)
        xb = X[:, :, j]
        if aug_shift > 0                              # 정수 채널 굴리기: 영점 이동 증강
            k = rand(rng, -round(Int, aug_shift / 0.1):round(Int, aug_shift / 0.1))
            xb = circshift(xb, (k, 0, 0))
        end
        Flux.adjust!(opt, onecycle_lr(2e-3, step, total)); step += 1
        g = only(Flux.gradient(m -> Flux.logitcrossentropy(m(xb), Flux.onehotbatch(y[j], 0:9)), net))
        Flux.update!(opt, net, g)
    end
    return net
end
acc(net, X) = mean(Flux.onecold(net(X), 0:9) .== yte)

function equivariance_check()
    rng = Xoshiro(0)
    conv = Conv((9,), 1 => 8; pad = 0, init = Flux.glorot_uniform(rng))
    cconv(x) = conv(NNlib.pad_circular(x, (4, 4); dims = 1))          # circular padding 4
    x0 = Xtr[:, :, 1:4]
    e_conv = maximum(abs.(cconv(circshift(x0, (7, 0, 0))) .- circshift(cconv(x0), (7, 0, 0))))
    pool = MaxPool((2,))
    e_even = maximum(abs.(pool(circshift(x0, (8, 0, 0))) .- circshift(pool(x0), (4, 0, 0))))
    e_odd = maximum(abs.(pool(circshift(x0, (7, 0, 0))) .- circshift(pool(x0), (3, 0, 0))))
    @printf("equivariance: |conv(shift x) - shift conv(x)|max = %.1e (circular, shift 7)\n", e_conv)
    @printf("   maxpool(2): shift 8 -> 4: %.1e;  shift 7 (odd) -> best integer shift 3: %.2f\n", e_even, e_odd)
end

function main()
    equivariance_check()
    shifts = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    Xte = Dict(s => test_set(s) for s in shifts)
    models = ["MLP" => (r -> MLPx(r), 0.0), "CNN+flatten" => (r -> CNN(r; head = :flat), 0.0),
              "CNN+global pool" => (r -> CNN(r), 0.0), "deep CNN+global pool" => (r -> CNN(r; deep = true), 0.0),
              "MLP + shift augmentation" => (r -> MLPx(r), 1.0)]
    println("model                         #params  " * join([@sprintf("%7.1f", s) for s in shifts]))
    for (name, (mk, aug)) in models
        A = zeros(2, length(shifts)); net = nothing
        for seed in 0:1
            net = train(mk(Xoshiro(seed)), Xtr, ytr; seed, aug_shift = aug)
            A[seed+1, :] = [acc(net, Xte[s]) for s in shifts]
        end
        println(rpad(name, 28), lpad(nparams(net), 8), " ", join([@sprintf("%7.3f", a) for a in vec(mean(A; dims = 1))]))
        if name == "deep CNN+global pool"
            pred = Flux.onecold(net(Xte[0.0]), 0:9); C = zeros(Int, 10, 10)
            for (t, p) in zip(yte, pred)
                C[t+1, p+1] += 1
            end
            println("   confusion (rows true, cols pred; seed 1, no shift): ", join(first.(SYS, 4), " "))
            for i in 1:10
                println("   ", rpad(first(SYS[i], 12), 12), join(lpad.(C[i, :], 5)))
            end
        end
    end
end

main()
