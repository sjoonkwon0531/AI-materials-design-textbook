# lab_7_16.jl — Lab 7.16: 에탄올에서 사전학습의 이득을 정직하게 재기
# (labs/74_pretrain_ethanol.py, 74_scratch_control.py, 74_long_finetune.py를 한 파일로)
# 모형: 밀집 SchNet(F 32, 상호작용 2, RBF 16, r_c 5 Å), E = scale · Σ_i ε_i / n + shift, 판독 마지막 층 0 초기화.
# 사전학습 (S) 좌표 잡음 제거(레이블 없음): x̃ = x + σε, -∇E(x̃)가 -ε를 예측. (G) 싼 레이블(GFN1-xTB)로 지도 학습.
# 미세조정 800단계(그리고 대조군: 처음부터 2,300단계, 사전학습 모형의 2,300단계 미세조정). 학습이 길어
# 파이썬 판과 같은 수십 분 규모다. 숫자는 시드 두 개의 평균으로 통계적으로 비교한다.
include("ch7_common.jl")
include("mlip_common.jl")

newmodel(Z) = DenseSchNet(Z .+ 1; nf = 32, nrbf = 16, rc = 5.0, ninter = 2, maxz = 10, pernorm = true)
function init_params(model, seed)
    ps, st = Lux.setup(Xoshiro(seed), model)
    fill!(ps.head.layer_2.weight, 0); fill!(ps.head.layer_2.bias, 0)   # 처음에는 E = shift
    return ps, st
end

"Adam + cosine(η_min = 0.02 η), 복원 추출 배치. denoise_sigma가 있으면 레이블 없는 잡음 제거 손실"
function train(model, ps, st, R, E, F, steps; lr = 2e-3, bs = 32, seed = 0, wE = 0.01, wF = 1.0, denoise_sigma = nothing)
    rng = Xoshiro(seed); n = size(R, 3)
    sE = E === nothing ? 1.0f0 : Float32(std(E)); sF = F === nothing ? 1.0f0 : Float32(std(F))
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    for s in 1:steps
        idx = rand(rng, 1:n, min(bs, n))
        Optimisers.adjust!(opt, cosine_lr(lr, s - 1, steps; eta_min = 0.02lr))
        g = if denoise_sigma !== nothing
            eps = randn(rng, Float32, 3, size(R, 2), length(idx))
            Rn = R[:, :, idx] .+ Float32(denoise_sigma) .* eps
            only(Zygote.gradient(p -> mean(abs2, energy_forces(model, p, st, Rn)[2] .+ eps), ps))
        else
            Rb, Eb, Fb = R[:, :, idx], E[idx], F[:, :, idx]
            only(Zygote.gradient(ps) do p
                Ep, Fp = energy_forces(model, p, st, Rb)
                wE * mean(abs2, (Ep .- Eb) ./ sE) + wF * mean(abs2, (Fp .- Fb) ./ sF)
            end)
        end
        opt, ps = Optimisers.update!(opt, ps, g)
    end
    return ps
end

function main(; long_runs = true)
    R3, Z, E3, F3 = load_ethanol(300)
    R3, E3, F3 = Float32.(R3), Float32.(E3), Float32.(F3)
    pool, test = 1:2000, 2001:3000
    g1 = npzread(joinpath(DATA, "74_ethanol_gfn1_600_1000K.npz"))
    Rp = Float32.(permutedims(g1["R"], (3, 2, 1))); E1 = Float32.(g1["E1"]); F1 = Float32.(permutedims(g1["F1"], (3, 2, 1)))
    @printf("pretraining geometries: %d; 300 K pool %d, test %d\n", size(Rp, 3), length(pool), length(test))
    model = newmodel(Z)
    pre = Dict{String,Any}()
    ps0, st = init_params(model, 0); pre["0"] = deepcopy(ps0)
    model.scale[] = 9.0; model.e0[] = 0.0
    pre["S"] = train(model, deepcopy(ps0), st, Rp, nothing, nothing, 1500; seed = 0, denoise_sigma = 0.05)
    model.scale[] = 9 * std(F1); model.e0[] = mean(E1)
    pre["G"] = train(model, deepcopy(ps0), st, Rp, E1, F1, 1500; seed = 0)
    Ez = vcat([first(model(R3[:, :, test[k:min(k + 249, end)]], pre["G"], st)) for k in 1:250:1000]...)
    dE = (Ez .- mean(Ez)) .- (E3[test] .- mean(E3[test]))                 # GFN1과 GFN2 에너지는 상수만큼 다르다
    _, fG = evaluate(model, pre["G"], st, R3[:, :, test], Float64.(E3[test]), F3[:, :, test])
    @printf("G zero-shot on GFN2 300 K test: E MAE (mean offset removed) %.0f meV, F MAE %.0f meV/A\n", mean(abs.(dE)) * 1e3, fG * 1e3)
    scale_of = Dict("0" => tr -> 9 * std(F3[:, :, tr]), "S" => tr -> 9.0, "G" => tr -> 9 * std(F1))
    function run(N, init, steps, seed)
        tr = read_idx("pretrain_N$(N)_seed$(seed).csv")                    # 파이썬 np.random.default_rng(100 + seed).choice(pool, N)
        model.e0[] = mean(E3[tr]); model.scale[] = scale_of[init](tr)
        ps = train(model, deepcopy(pre[init]), st, R3[:, :, tr], E3[tr], F3[:, :, tr], steps; seed)
        eE, eF = evaluate(model, ps, st, R3[:, :, test], Float64.(E3[test]), F3[:, :, test])
        return eE * 1e3, eF * 1e3
    end
    for N in (50, 200), init in ("0", "S", "G")
        r = [run(N, init, 800, s) for s in 0:1]
        @printf("N=%4d init=%s: E MAE %6.1f meV   F MAE %6.1f meV/A   (800 fine-tuning steps)\n", N, init, mean(first.(r)), mean(last.(r)))
    end
    long_runs || return
    for N in (50, 200), init in ("0", "S", "G")                            # 대조: 같은 총 단계(1,500 + 800 = 2,300)
        r = [run(N, init, 2300, s) for s in 0:1]
        @printf("N=%4d init=%s, 2300 steps: E MAE %6.1f meV   F MAE %6.1f meV/A\n", N, init, mean(first.(r)), mean(last.(r)))
    end
end

main()
