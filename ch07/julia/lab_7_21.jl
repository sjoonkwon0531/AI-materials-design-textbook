# lab_7_21.jl — Lab 7.21: 같은 데이터, 같은 예산, 다섯 모델 (labs/77_models.py, 77_train.py)
# DirectMLP(원좌표), DirectMLP + 회전 증강, DirectMLP(분자 좌표계로 정준화), SchNetLike(F 64, 상호작용 3),
# NequIP형(equiv_layers.jl, 16x0e + 8x1o + 8x1e, l_max 1, 2층)을 N = 100, 300, 1000에서 같은 단계 예산으로 학습하고
# 300 K 시험, 600/1000 K 외삽, 무작위 회전 시험, 회전 등변 오차를 잰다. 모든 모형은 F = -∂E/∂R로 보존적이다.
# 학습 15번이 CPU 한 스레드에서 파이썬과 비슷한 수십 분 규모다(등변 모형이 가장 느리다).
include("ch7_common.jl")
include("mlip_common.jl")
include("equiv_layers.jl")
include("models77.jl")                                                   # DirectMLP, quat_rotations, build

"Adam + cosine(η_min = 0.01 η), 복원 추출 배치, 손실 λ_E ΔE² + |ΔF|²; 예산의 1/20마다 검증 F_MAE + 0.1 E_MAE/9로 최선 보관"
function train77(kind, Z, Rtr, Etr, Ftr, Rva, Eva, Fva; steps, bs, lr, seed = 0)
    rng = Xoshiro(seed)
    model = build(kind, Z); ps, st = Lux.setup(rng, model)
    model.e0[] = mean(Etr); model.scale[] = sqrt(mean(abs2, Ftr))           # 힘 RMS가 에너지 척도를 정한다(NequIP식)
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    best, bestps, n = Inf, deepcopy(ps), length(Etr)
    for s in 1:steps
        idx = rand(rng, 1:n, min(bs, n))
        R, E, F = Rtr[:, :, idx], Etr[idx], Ftr[:, :, idx]
        if kind == :mlp_aug                                                 # 입력과 힘 레이블을 함께 회전
            Q = quat_rotations(rng, length(idx)); R = rotate_batch(Q, R); F = rotate_batch(Q, F)
        end
        Optimisers.adjust!(opt, cosine_lr(lr, s - 1, steps; eta_min = 0.01lr))
        g = only(Zygote.gradient(ps) do p
            Ep, Fp = energy_forces(model, p, st, R)
            mean(abs2, Ep .- E) + mean(abs2, Fp .- F)
        end)
        opt, ps = Optimisers.update!(opt, ps, g)
        if s % max(steps ÷ 20, 1) == 0 || s == steps
            eE, eF = evaluate(model, ps, st, Rva, Eva, Fva; bs = 100)
            sc = eF + 0.1 * eE / 9
            sc < best && (best = sc; bestps = deepcopy(ps))
        end
    end
    return model, bestps, st
end

function main()
    R3, Z, E3, F3 = load_ethanol(300); R6, _, E6, F6 = load_ethanol(600); R10, _, E10, F10 = load_ethanol(1000)
    R3, F3 = Float32.(R3), Float32.(F3)
    perm = read_idx("ethanol_pool_perm.csv")                               # np.random.default_rng(0).permutation(2000)
    val, test = 2001:2100, 2201:3000
    i6, i10 = sort(randperm(Xoshiro(1), 3000)[1:500]), sort(randperm(Xoshiro(2), 1000)[1:500])   # 파이썬과 다른 표본(통계적 비교)
    STEPS = Dict(100 => 800, 300 => 1200, 1000 => 2000)
    for N in (100, 300, 1000), kind in (:mlp, :mlp_aug, :mlp_frame, :schnet, :nequip)
        tr = perm[1:N]; mlp = kind in (:mlp, :mlp_aug, :mlp_frame)
        cpu = @elapsed (model, ps, st) = train77(kind, Z, R3[:, :, tr], Float32.(E3[tr]), F3[:, :, tr], R3[:, :, val], E3[val], F3[:, :, val];
                                                  steps = mlp ? 4000 : STEPS[N], bs = mlp ? 32 : 8, lr = mlp ? 1e-3 : 1e-2)
        r3 = evaluate(model, ps, st, R3[:, :, test], E3[test], F3[:, :, test])
        r6 = evaluate(model, ps, st, R6[:, :, i6], E6[i6], F6[:, :, i6]); r10 = evaluate(model, ps, st, R10[:, :, i10], E10[i10], F10[:, :, i10])
        Q = quat_rotations(Xoshiro(7), length(test))
        rr = evaluate(model, ps, st, rotate_batch(Q, R3[:, :, test]), E3[test], rotate_batch(Q, F3[:, :, test]))
        Q2 = quat_rotations(Xoshiro(3), 200); Rs = R3[:, :, test[1:200]]
        E0, F0 = energy_forces(model, ps, st, Rs); E1, F1 = energy_forces(model, ps, st, rotate_batch(Q2, Rs))
        rotF = mean(abs.(F1 .- rotate_batch(Q2, F0))) / mean(abs.(F0)); rotE = mean(abs.(E1 .- E0))
        @printf("N=%4d %-9s params %6d | 300K: E %6.1f meV F %5.0f meV/A | 600K F %5.0f | 1000K F %5.0f | rotated 300K F %5.0f | rot.err F %.1e E %.1e meV | %4.0f s\n",
                N, kind, count_params(ps), r3[1] * 1e3, r3[2] * 1e3, r6[2] * 1e3, r10[2] * 1e3, rr[2] * 1e3, rotF, rotE * 1e3, cpu)
        N == 1000 && kind in (:schnet, :nequip, :mlp, :mlp_aug, :mlp_frame) && serialize_params(kind, ps)
    end
end

"Lab 7.22에서 쓸 매개변수 저장 (Julia 표준 Serialization)"
function serialize_params(kind, ps)
    open(joinpath(@__DIR__, "77_$(kind)_N1000.jls"), "w") do io
        Serialization.serialize(io, ps)
    end
end
import Serialization

main()
