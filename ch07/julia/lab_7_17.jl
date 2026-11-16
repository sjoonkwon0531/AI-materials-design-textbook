# lab_7_17.jl — Lab 7.17: LoRA를 처음부터, 밴드갭 MLP를 새 화학 영역에 맞추기 (labs/75_lora_bandgap.py)
# 원천 영역: N, P, As, Sb가 없는 화합물. 목표 영역: 그 원소를 하나라도 가진 화합물(968개; 화학식 정규식으로 판정,
# pymatgen Composition과 4,601개 모두 같은 판정). MLP 132-256-256-128-1(SiLU)을 원천에서 사전학습하고
# 처음부터 / 영점 / 선형 탐침 / 전체 미세조정 / LoRA r = 1, 2, 4, 8 / 병목 어댑터를 비교한다.
# Flux.@layer의 trainable 지정으로 LoRA의 A, B만 학습되게 하고, 동결은 Optimisers.freeze!로 한다.
include("ch7_common.jl")
include("flux_tools.jl")

"y = W x + b + (α/r) B A x; W, b는 동결, A ~ N(0, 1/in), B = 0 (시작할 때 ΔW = 0)"
struct LoRA{L,M}
    lin::L
    A::M
    B::M
    s::Float32
end
Flux.@layer LoRA trainable = (A, B)
LoRA(lin::Dense, r; alpha = r) = LoRA(lin, randn(Float32, r, size(lin.weight, 2)) ./ sqrt(Float32(size(lin.weight, 2))),
                                      zeros(Float32, size(lin.weight, 1), r), Float32(alpha / r))
(l::LoRA)(x) = l.lin(x) .+ l.s .* (l.B * (l.A * x))
delta(l::LoRA) = l.s .* (l.B * l.A)
"어댑터를 끈 모형: LoRA를 원래 Dense로 되돌린다(사전학습 모형과 정확히 같아야 한다)"
strip_lora(m::Chain) = Chain([l isa LoRA ? l.lin : l for l in m.layers]...)

"Houlsby 병목 어댑터: h + W_up f(W_down h), W_up = 0으로 시작(항등에 가깝다)"
struct Adapter{D,U}
    down::D
    up::U
end
Flux.@layer Adapter
Adapter(rng, d, m) = Adapter(tdense(rng, d, m), Dense(zeros(Float32, d, m), zeros(Float32, d)))
(a::Adapter)(h) = h .+ a.up(swish.(a.down(h)))

function main()
    X, y, formula = load_magpie()
    is_t = [!isempty(intersect(elements_of(f), Set(["N", "P", "As", "Sb"]))) for f in formula]
    src = read_idx("lora_src_perm.csv"); tgt = read_idx("lora_tgt_perm.csv")        # 파이썬 rng(0)의 순열 (없으면 아래 두 줄)
    # rng = Xoshiro(0); src = shuffle(rng, findall(.!is_t)); tgt = shuffle(rng, findall(is_t))
    src_tr, src_te = src[1:floor(Int, 0.8 * length(src))], src[floor(Int, 0.8 * length(src))+1:end]
    tgt_pool, tgt_te = tgt[1:400], tgt[401:end]
    mu = vec(mean(X[src_tr, :]; dims = 1)); sd = vec(std(X[src_tr, :]; dims = 1, corrected = false)) .+ 1e-6
    Xs = Float32.(permutedims((X .- mu') ./ sd')); ys = Float32.(y)          # 132 x n, 원천 학습 통계로만 표준화
    @printf("source train %d, test %d | target pool %d, test %d (pnictogen compounds: %d)\n",
            length(src_tr), length(src_te), length(tgt_pool), length(tgt_te), count(is_t))
    mae_on(m, idx) = mean(abs.(vec(m(Xs[:, idx])) .- ys[idx]))
    mlp(rng) = mlp_chain(rng, [132, 256, 256, 128, 1])

    # ---- 원천 사전학습: AdamW(2e-3, wd 1e-4) + cosine, 3000단계, 배치 128(복원 추출), L1 손실
    rng = Xoshiro(0); base = mlp(rng)
    opt = Flux.setup(TorchAdamW(2e-3; lambda = 1e-4), base)
    for s in 1:3000
        Flux.adjust!(opt, cosine_lr(2e-3, s - 1, 3000))
        b = src_tr[rand(rng, 1:length(src_tr), 128)]
        g = only(Flux.gradient(m -> mean(abs.(vec(m(Xs[:, b])) .- ys[b])), base))
        Flux.update!(opt, base, g)
    end
    @printf("pretrained: params %d, source-test MAE %.3f, zero-shot target-test MAE %.3f\n", nparams(base), mae_on(base, src_te), mae_on(base, tgt_te))

    # 방법별 모형과 동결할 Dense 층 번호
    function build(method, seed)
        rng = Xoshiro(1000 + seed)
        method == "scratch" && return mlp(rng), Int[]
        m = deepcopy(base)
        method == "full" && return m, Int[]
        method == "zero-shot" && return m, [1, 3, 5, 7]                   # Dense 층 번호 (2, 4, 6은 활성화)
        method == "linear probe" && return m, [1, 3, 5]
        if startswith(method, "LoRA")
            r = parse(Int, split(method, "=")[2])
            return Chain([l isa Dense ? LoRA(l, r) : l for l in m.layers]...), Int[]      # Dense는 LoRA 안에서 동결
        end
        L = m.layers                                                       # 어댑터: 바탕 Dense 4개(1, 4, 7, 10번)는 동결
        return Chain(L[1], L[2], Adapter(rng, 256, 8), L[3], L[4], Adapter(rng, 256, 8), L[5], L[6], Adapter(rng, 128, 8), L[7]), [1, 4, 7, 10]
    end
    ntrain(m, frozen) = sum(length, Flux.trainables(m)) - sum(i -> sum(length, Flux.trainables(m.layers[i]); init = 0), frozen; init = 0)

    # 전체 배치 Adam(1e-3) L1, 최대 400단계; N개 중 20%로 조기 종료(10단계마다, 시작점 포함). 시험 세트는 보지 않는다
    function finetune(m, frozen, idx, seed; steps = 400, lr = 1e-3)
        perm = shuffle(Xoshiro(10 + seed), idx); nv = max(length(idx) ÷ 5, 5)
        va, tr = perm[1:nv], perm[nv+1:end]
        ntrain(m, frozen) == 0 && return m
        opt = Flux.setup(Optimisers.Adam(lr), m)
        for i in frozen
            Optimisers.freeze!(opt.layers[i])
        end
        best = (mae_on(m, va), deepcopy(m))
        for s in 1:steps
            g = only(Flux.gradient(mm -> mean(abs.(vec(mm(Xs[:, tr])) .- ys[tr])), m))
            Flux.update!(opt, m, g)
            if s % 10 == 0
                v = mae_on(m, va)
                v < best[1] && (best = (v, deepcopy(m)))
            end
        end
        return best[2]
    end

    methods = ["scratch", "zero-shot", "linear probe", "full", "LoRA r=1", "LoRA r=2", "LoRA r=4", "LoRA r=8", "adapter"]
    for N in (50, 200)
        println("--- N = $N labelled target compounds (mean +- sd over 5 subsets) ---")
        for meth in methods
            tm, sm, npar = Float64[], Float64[], 0
            for seed in 0:4
                idx = read_idx("lora_N$(N)_seed$(seed).csv")             # 파이썬 default_rng(seed).choice(tgt_pool, N)
                m0, frozen = build(meth, seed)
                npar = ntrain(m0, frozen)
                m = finetune(m0, frozen, idx, seed)
                push!(tm, mae_on(m, tgt_te)); push!(sm, mae_on(m, src_te))
                if meth == "full" && N == 200 && seed == 0
                    dW = m.layers[3].weight - base.layers[3].weight
                    sv = svdvals(dW); e = sv .^ 2 ./ sum(sv .^ 2)
                    @printf("    Delta W (layer 2, 256x256): top-5 sv %s, s1/s10 = %.1f, energy top 1/4/8: %.2f/%.2f/%.2f, ||dW||/||W|| = %.4f\n",
                            round.(sv[1:5]; digits = 4), sv[1] / sv[10], sum(e[1:1]), sum(e[1:4]), sum(e[1:8]), norm(dW) / norm(base.layers[3].weight))
                end
                if meth == "LoRA r=4" && N == 200 && seed == 0
                    @printf("    LoRA r=4 with adapters switched off: source-test MAE %.3f (pretrained %.3f)\n", mae_on(strip_lora(m), src_te), mae_on(base, src_te))
                end
            end
            @printf("  %-13s trainable %7d  target-test MAE %.3f +- %.3f   source-test MAE %.3f\n", meth, npar, mean(tm), std(tm; corrected = false), mean(sm))
        end
    end
end

main()
