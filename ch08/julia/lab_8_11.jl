# lab_8_11.jl — Lab 8.11: 입력 공간의 강건성과 날카로움의 관계 (labs/83_input_robust.py)
# Lab 8.9가 저장한 35개 모형(7가지 x 시드 5개)을 읽는다. FROM_PYTHON = true면 파이썬이 저장한 .pt를 PythonCall로 옮겨 쓴다
# (그러면 모형 자체가 같아 (1)의 숫자가 파이썬과 같아야 한다; 이때 Lab 8.9의 결과표는 파이썬 json에서 와야 하므로 (2)의 상관은 생략).
const FROM_PYTHON = false
FROM_PYTHON && @eval using PythonCall
include("bandgap83.jl")

getm(tag, s) = FROM_PYTHON ? load_torch_mlp(tag, s) : load_model(tag, s)
Ztr = Float64.(Zf(TR_ALL))                                            # 특징 x 관측

# ---- (0) 분포 밖 탐지: 시험(ID) 대 산화물·할라이드
"학습 집합의 5-최근접 이웃까지의 평균 거리 (전수 계산)"
function knn_dist(idx; k = 5)
    Zq = Float64.(Zf(idx))
    D2 = sum(abs2, Zq; dims = 1)' .+ sum(abs2, Ztr; dims = 1) .- 2 .* (Zq' * Ztr)     # (질의, 학습) 거리²
    return [mean(sqrt.(max.(partialsort(D2[i, :], 1:k), 0))) for i in 1:size(D2, 1)]
end
const MU_Z = vec(mean(Ztr; dims = 2))
const PREC = pinv(cov(Ztr; dims = 2) + 1e-3 * I)
maha(idx) = (D = Float64.(Zf(idx)) .- MU_Z; vec(sqrt.(sum(D .* (PREC * D); dims = 1))))
ens = [getm("adamw", s) for s in 0:4]
enspred(idx) = reduce(vcat, [vec(m(Zf(idx)))' for m in ens])                     # 5 x n
spread(idx) = vec(std(enspred(idx); dims = 1, corrected = false))
emean(idx) = max.(vec(mean(enspred(idx); dims = 1)), 0)
ybar = mean(YG[TR_ALL])
@printf("(0) baseline 'predict the training mean %.3f eV': MAE ID %.3f, oxide %.3f, halide %.3f eV\n",
        ybar, mean(abs.(YG[TE] .- ybar)), mean(abs.(YG[OX] .- ybar)), mean(abs.(YG[HA] .- ybar)))
println("(0) OOD detection, ID test vs oxides / halides: AUROC")
for (nm, fn) in [("5-NN distance", knn_dist), ("Mahalanobis", maha), ("ensemble sd (5 AdamW)", spread)]
    a1 = auroc([falses(length(TE)); trues(length(OX))], [fn(TE); fn(OX)])
    a2 = auroc([falses(length(TE)); trues(length(HA))], [fn(TE); fn(HA)])
    @printf("   %-24s oxide %.3f  halide %.3f\n", nm, a1, a2)
end
for (nm, idx) in [("ID test", TE), ("oxide", OX)]
    err = abs.(emean(idx) .- YG[idx]); sd = spread(idx)
    @printf("   %-8s ensemble MAE %.3f eV, mean sd %.3f eV, coverage of mean +/- 1.96 sd %.1f%%\n", nm, mean(err), mean(sd), 100 * mean(err .<= 1.96 .* sd))
end

# ---- (1) 재매개변수화: ReLU 망에서 W1, b1 *= a, W2 /= a는 함수를 바꾸지 않지만 Hessian 고윳값을 바꾼다
m0 = getm("adamw", 0); Xt = Zf(TE); f0 = m0(Xt)
println("(1) reparameterisation of AdamW seed-0 model: W1,b1 *= a, W2 /= a  (exact HVP, power iteration run to convergence: 300 steps)")
for a in (1.0f0, 4.0f0, 0.25f0)
    m2 = deepcopy(m0); m2[1].weight .*= a; m2[1].bias .*= a; m2[3].weight ./= a
    @printf("   a=%-5g max|f'-f| = %.1e eV | lambda_max(H) = %8.2f | lambda_max(THT) = %.4f\n",
            a, maximum(abs.(m2(Xt) .- f0)), sharpness(m2, SUB; iters = 300, tol = 0.0), sharpness(m2, SUB; adaptive = true, iters = 300, tol = 0.0))
end

# ---- (2) 입력 공간 강건성
TAGS = [("AdamW (bs 64)", "adamw"), ("AdamW (bs 1024)", "adamw_bs"), ("SAM", "sam"), ("ASAM", "asam"),
        ("F-SAM", "fsam"), ("AdamW + input noise", "adamw_noise"), ("AdamW + FGSM", "adamw_fgsm")]
function metrics(model, idx; seed = 0)
    X, y = Zf(idx), YG[idx]; rng = Xoshiro(seed)
    gx = only(Zygote.gradient(x -> sum(model(x)), X))                          # 표본별 ∂f/∂x (열)
    out = Dict{String,Float64}("gnorm" => mean(norm.(eachcol(gx))))
    base = gap_mae(model, X, y)
    for s in (0.1, 0.3)
        out["noise$s"] = mean(gap_mae(model, X .+ Float32(s) .* randn(rng, Float32, size(X)), y) for _ in 1:5) - base
    end
    ga = only(Zygote.gradient(x -> sum(abs.(vec(model(x)) .- y)), X))           # L1 손실의 입력 기울기
    for e in (0.05, 0.1)
        out["fgsm$e"] = gap_mae(model, X .+ Float32(e) .* sign.(ga), y) - base
    end
    return out
end
println("(2) input-space robustness on the ID test set, increase in MAE [eV]; mean over 5 seeds")
R89 = FROM_PYTHON ? nothing : deserialize(joinpath(MODELDIR, "83_bandgap_results.jls"))
allr = Vector{Float64}[]
for (name, tag) in TAGS
    M = [metrics(getm(tag, s), TE; seed = s) for s in 0:4]
    A = k -> [x[k] for x in M]
    lamtxt = R89 === nothing ? "" : @sprintf(" | lam_max %.1f", mean(r.lam for r in R89[name]))
    @printf("   %-20s |grad_x f| %.3f+/-%.3f | noise 0.1: +%.3f  0.3: +%.3f | FGSM 0.05: +%.3f  0.1: +%.3f%s\n", name,
            mean(A("gnorm")), std(A("gnorm"); corrected = false), mean(A("noise0.1")), mean(A("noise0.3")),
            mean(A("fgsm0.05")), mean(A("fgsm0.1")), lamtxt)
    R89 === nothing && continue
    for s in 1:5
        r = R89[name][s]; push!(allr, [r.lam, r.lam_ad, M[s]["gnorm"], M[s]["fgsm0.1"], r.id, r.ox])
    end
end
if !isempty(allr)
    Ar = reduce(hcat, allr)'; lab = ["lam_max", "lam(THT)", "|grad_x f|", "FGSM0.1 dMAE", "ID MAE", "oxide MAE"]
    println("Spearman rank correlations over 35 models:")
    for (i, j) in ((1, 3), (1, 4), (1, 5), (1, 6), (2, 5), (2, 6), (3, 6), (5, 6))
        @printf("   %-12s vs %-12s rho = %+.2f\n", lab[i], lab[j], spearman(Ar[:, i], Ar[:, j]))
    end
end
