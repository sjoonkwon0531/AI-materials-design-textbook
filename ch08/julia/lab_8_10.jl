# lab_8_10.jl — Lab 8.10: 1000 K 시험, SAM은 MLIP의 온도 외삽을 돕는가 (labs/83_mlip_sam.py)
# 모형은 7장 Lab 7.21의 SchNet형(밀집 형식, F = 64, RBF 32, 상호작용 3층; 7장 mlip_common.jl의 DenseSchNet)이다.
# 손실에 힘이 들어가므로 매개변수 기울기는 Lux의 중첩 AD(7장과 같은 energy_forces)로 구한다. 매개변수는 ComponentArray로
# 평평하게 만들어 sam.jl의 SAM·ASAM 갱신과 λ_max를 그대로 쓴다. 손실에 이미 중첩 AD(힘)가 있으므로 λ_max의 HVP만은
# 중심 차분(hvp_fd)으로 둔다. 활성화(shifted softplus)와 절단이 매끄러워 ReLU 망과 달리 차분 오차가 ~h²로 작다.
# 무거운 Lab이다. 파이썬 판은 학습 12회(ρ 고르기 3 + 세 방법 x 시드 3)에 약 20분이 걸렸다. Julia 판도 같은 규모의 계산이다.
include("ch8_common.jl")
include(joinpath(CH7_JL, "ch7_common.jl"))         # DATA(7장 data/), read_idx, cosine_lr
include(joinpath(CH7_JL, "mlip_common.jl"))        # load_ethanol, DenseSchNet, energy_forces, evaluate (Lux)
include("sam.jl")
using ComponentArrays

R3, Z, E3, F3 = load_ethanol(300); R6, _, E6, F6 = load_ethanol(600); R10, _, E10, F10 = load_ethanol(1000)
R3, F3 = Float32.(R3), Float32.(F3)
perm = read_idx("ethanol_pool_perm.csv")                          # np.random.default_rng(0).permutation(2000)
tr, val, test = perm[1:300], 2001:2100, 2201:3000
i6, i10 = isfile(joinpath(CH8_DATA, "splits", "mlip_i6.csv")) ? (read_idx8("mlip_i6.csv"), read_idx8("mlip_i10.csv")) :
          (sort(randperm(Xoshiro(1), 3000)[1:500]), sort(randperm(Xoshiro(2), 1000)[1:500]))   # 없으면 다른 표본(통계적 비교)
Rtr, Etr, Ftr = R3[:, :, tr], Float32.(E3[tr]), F3[:, :, tr]

function build_model()
    sp = Dict(1 => 1, 6 => 2, 8 => 3)
    m = DenseSchNet([sp[z] for z in Z]; nf = 64, nrbf = 32, rc = 5.0, ninter = 3, maxz = 3, gamma = 10.0)
    m.e0[] = mean(Etr); m.scale[] = sqrt(mean(abs2, Ftr))           # E = μ + σ Σ_i ε_i (파이썬 Model과 같음)
    return m
end
"손실 <ΔE²> + <ΔF²> (성분 평균)"
ef_loss(model, st, R, E, F) = p -> begin
    Ep, Fp = energy_forces(model, p, st, R)
    mean(abs2, Ep .- E) + mean(abs2, Fp .- F)
end

"method = :none(Adam), :sam, :asam. 배치 8(복원 추출), 1200 단계, Adam lr 1e-2, cosine(η_min = lr/100)"
function train(method, seed; rho = 0.05, steps = 1200, bs = 8, lr = 1e-2)
    rng = Xoshiro(seed); model = build_model()
    ps, st = Lux.setup(rng, model); θ = ComponentArray(ps)
    opt = Optimisers.setup(Optimisers.Adam(lr), θ); ss = SAMState()
    for s in 0:steps-1
        idx = rand(rng, 1:length(Etr), bs)
        Optimisers.adjust!(opt, cosine_lr(lr, s, steps; eta_min = lr * 0.01))
        opt, θ, _ = sam_update!(opt, θ, ef_loss(model, st, Rtr[:, :, idx], Etr[idx], Ftr[:, :, idx]), ss; rho, mode = method)
    end
    return model, θ, st
end
"처음 64개 학습 구조 위 손실의 λ_max (Float64 사본, 25회, 차분 HVP)"
function lam(model, θ, st; n = 64)
    j = perm[1:n]
    L = ef_loss(model, st, Float64.(R3[:, :, j]), E3[j], Float64.(F3[:, :, j]))
    return lambda_max(L, Float64.(θ); iters = 25, hvp = hvp_fd)[1]
end
ev(model, θ, st, R, E, F) = evaluate(model, θ, st, R, E, F; bs = 100)

sel = Dict{Float64,Float64}()
for r in (0.01, 0.05, 0.2)                                        # SAM의 ρ는 검증 프레임으로 (시드 0)
    m, θ, st = train(:sam, 0; rho = r); sel[r] = ev(m, θ, st, R3[:, :, val], E3[val], F3[:, :, val])[2]
    @printf("  SAM rho=%g: val force MAE %.0f meV/A\n", r, 1e3 * sel[r])
end
rho = argmin(sel)                                                 # Dict에 argmin: 값이 가장 작은 키
for (name, m_, r) in [("Adam", :none, 0.0), ("SAM", :sam, rho), ("ASAM", :asam, 10 * rho)]   # ASAM은 상대 반경이라 10배
    rows = []
    for seed in 0:2
        local model, θ, st
        cpu = @elapsed ((model, θ, st) = train(m_, seed; rho = r))
        e3, f3 = ev(model, θ, st, R3[:, :, test], E3[test], F3[:, :, test])
        f6 = ev(model, θ, st, R6[:, :, i6], E6[i6], F6[:, :, i6])[2]; f10 = ev(model, θ, st, R10[:, :, i10], E10[i10], F10[:, :, i10])[2]
        push!(rows, (F300 = f3, F600 = f6, F1000 = f10, lam = lam(model, θ, st)))
        @printf("   %-5s seed %d: E300 %5.1f meV | F300 %4.0f F600 %4.0f F1000 %4.0f meV/A | lam_max %8.1f | %4.0f s\n",
                name, seed, 1e3 * e3, 1e3 * f3, 1e3 * f6, 1e3 * f10, rows[end].lam, cpu)
    end
    A = k -> [getfield(x, k) for x in rows]
    @printf("%-5s (rho %g) F300 %4.0f+/-%2.0f  F600 %4.0f+/-%2.0f  F1000 %4.0f+/-%2.0f meV/A  ratio 1000/300 %.2f  lam_max %.0f+/-%.0f\n",
            name, r, 1e3 * mean(A(:F300)), 1e3 * std(A(:F300); corrected = false), 1e3 * mean(A(:F600)), 1e3 * std(A(:F600); corrected = false),
            1e3 * mean(A(:F1000)), 1e3 * std(A(:F1000); corrected = false), mean(A(:F1000) ./ A(:F300)), mean(A(:lam)), std(A(:lam); corrected = false))
end
