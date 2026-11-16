# lab_8_9.jl — Lab 8.9: 밴드갭 MLP 위의 AdamW, SAM, ASAM, F-SAM. 평평함은 분포 밖 오차를 줄이는가 (labs/83_bandgap_sam.py)
# Flux.jl(149-256-256-1, ReLU) + Zygote + Optimisers. SAM 계열은 sam.jl의 '두 번 계산하는 갱신'(평평한 θ 위의 Optimisers 상태).
# 파이썬 판과 같은 규모의 계산이다(ρ 고르기 9회 + 7가지 x 시드 5개 = 학습 44회, 각각 λ_max 두 번).
# 학습한 모형은 julia/models/에 저장하고 Lab 8.11이 읽는다.
include("bandgap83.jl")

@printf("n: train %d (of which val %d), ID test %d, OOD oxide %d, halide %d\n", length(TR_ALL), length(VA), length(TE), length(OX), length(HA))
@printf("mean gap: train %.3f  ID %.3f  oxide %.3f  halide %.3f eV; metals in train %.0f%%\n",
        mean(YF[TR_ALL]), mean(YF[TE]), mean(YF[OX]), mean(YF[HA]), 100 * mean(YF[TR_ALL] .== 0))

# ---- (1) 검증 집합으로 ρ 고르기 (시드 0)
Xtr, ytr = Zf(TR), YG[TR]
RHO = [:sam => (0.02, 0.05, 0.2), :asam => (0.2, 0.5, 2.0), :fsam => (0.02, 0.05, 0.2)]
best_rho = Dict{Symbol,Float64}()
for (m, grid) in RHO
    scores = [mae_on(fit(Xtr, ytr, m; seed = 0, rho = r), VA) for r in grid]
    best_rho[m] = grid[argmin(scores)]
    println("  rho selection $m: ", join([@sprintf("rho=%g val MAE %.3f", r, s) for (r, s) in zip(grid, scores)], "  "), "  -> $(best_rho[m])")
end

# ---- (2) train+val로 다시 학습, 시드 5개
Xtr, ytr = Zf(TR_ALL), YG[TR_ALL]
METHODS = [("AdamW (bs 64)", :none, "adamw", (;)), ("AdamW (bs 1024)", :none, "adamw_bs", (bs = 1024, lr = 4e-3)),
           ("SAM", :sam, "sam", (rho = best_rho[:sam],)), ("ASAM", :asam, "asam", (rho = best_rho[:asam],)),
           ("F-SAM", :fsam, "fsam", (rho = best_rho[:fsam],)),
           ("AdamW + input noise", :none, "adamw_noise", (noise = 0.1,)), ("AdamW + FGSM", :none, "adamw_fgsm", (fgsm = 0.05,))]
results = Dict{String,Vector{NamedTuple}}()
for (name, m, tag, kw) in METHODS
    rows = NamedTuple[]
    for seed in 0:4
        local mod
        cpu = @elapsed (mod = fit(Xtr, ytr, m; seed, kw...))
        push!(rows, (id = mae_on(mod, TE), ox = mae_on(mod, OX), ha = mae_on(mod, HA), train = mae_on(mod, TR_ALL),
                     lam = sharpness(mod, SUB), lam_ad = sharpness(mod, SUB; adaptive = true), cpu = cpu))
        save_model(tag, seed, mod)
    end
    results[name] = rows
    A = k -> [getfield(r, k) for r in rows]
    @printf("%-20s train %.3f | ID %.3f+/-%.3f | oxide %.3f+/-%.3f | halide %.3f+/-%.3f | lam_max %7.1f+/-%5.1f | lam(THT) %6.2f+/-%5.2f | %3.0f s/model\n",
            name, mean(A(:train)), mean(A(:id)), std(A(:id); corrected = false), mean(A(:ox)), std(A(:ox); corrected = false),
            mean(A(:ha)), std(A(:ha); corrected = false), mean(A(:lam)), std(A(:lam); corrected = false),
            mean(A(:lam_ad)), std(A(:lam_ad); corrected = false), mean(A(:cpu)))
end
serialize(joinpath(MODELDIR, "83_bandgap_results.jls"), results)
