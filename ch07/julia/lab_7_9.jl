# lab_7_9.jl — Lab 7.9: SchNet 대 ACSF 신경망, 에탄올의 에너지와 힘 (labs/72b_mlip.py)
# 같은 데이터·같은 손실(0.1 <ΔE²> + <|ΔF|²>/3)·같은 최적화기(Adam 3e-3 + cosine, 4000 단계, 배치 16)로
# 밀집 SchNet(F 32, RBF 20, 상호작용 3)과 ACSF + 원소별 MLP(BPNN, 은닉 48)를 학습한다. 힘을 손실에 넣으므로
# 매개변수 기울기는 2계 미분이다(Lux 중첩 AD). CPU 한 스레드에서 파이썬 판과 비슷하게 수십 분 걸린다.
include("ch7_common.jl")
include("mlip_common.jl")

R, Z, E, F = load_ethanol(300)
tr, va, te = split_300K()
@printf("ethanol 300 K: %d frames, E std %.0f meV, <|F_comp|> %.0f meV/A\n", length(E), std(E) * 1e3, mean(abs.(F)) * 1e3)
@printf("baseline (train-mean E, zero F): E MAE %.1f meV, F MAE %.1f meV/A\n",
        mae(E[te], mean(E[tr])) * 1e3, mean(abs.(F[:, :, te])) * 1e3)

res = Dict{Tuple{String,Int},Tuple{Float64,Float64}}()
for N in (100, 300, 1000), name in ("BPNN(ACSF)", "SchNet")
    idx = tr[1:N]
    model = name == "SchNet" ? DenseSchNet(Z .+ 1; nf = 32, nrbf = 20, rc = 5.0, ninter = 3) : fit_scaler!(BPNN(Z; hidden = 48), R[:, :, idx])
    ps, st = Lux.setup(Xoshiro(0), model)
    t = @elapsed ps = train_ef(model, ps, st, R[:, :, idx], E[idx], F[:, :, idx], R[:, :, va], E[va], F[:, :, va];
                                steps = 4000, bs = 16, lr = 3e-3)
    eE, eF = evaluate(model, ps, st, R[:, :, te], E[te], F[:, :, te])
    res[(name, N)] = (eE * 1e3, eF * 1e3)
    @printf("N=%5d %-11s params %6d  test E MAE %6.1f meV  F MAE %6.1f meV/A  (%.0f s)\n", N, name, count_params(ps), eE * 1e3, eF * 1e3, t)
end
for name in ("BPNN(ACSF)", "SchNet")
    Ns = [100, 300, 1000]; fm = [res[(name, n)][2] for n in Ns]
    @printf("%-11s force learning-curve slope d log MAE / d log N = %.2f\n", name, ([log.(Ns) ones(3)] \ log.(fm))[1])
end
