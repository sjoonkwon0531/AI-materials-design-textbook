# Lab 6.19 (Julia) — t와 tau를 576개 화합물에서 재현하기 (+ §6.8.3 Platt 보정)
# 파이썬 labs/68_perovskite.py의 앞부분 대응. 입력은 Bartel 등(2019) Table S1(labs/bartel_TableS1.csv).
# 모든 계산이 결정론적이므로 숫자가 그대로 일치해야 한다. 문턱 다시 맞추기는 sklearn과 같은 규칙(Gini, 중점 문턱)의
# 작은 CART(cart.jl)로 하고, Platt 보정은 C = 1e6의 로지스틱 회귀(Newton)로 한다.
# 예외: (t, mu) 깊이 3 트리의 학습 정확도는 동률 분기 처리 차이로 0.848(파이썬 0.850)이 될 수 있다(시험 0.862는 같다).
using Printf, Statistics, CSV, DataFrames
include("ch6_common.jl")
include("cart.jl")

function main()
    d = CSV.read(joinpath(LABS, "bartel_TableS1.csv"), DataFrame)
    rename!(d, "rA (Ang)" => "rA", "rB (Ang)" => "rB", "rX (Ang)" => "rX")
    y = Int.(d.exp_label .> 0); tr = d.is_train .== 1; te = d.is_train .== -1
    @printf("compounds %d: perovskite %d, non-perovskite %d; train %d / test %d\n", nrow(d), sum(y), sum(1 .- y), sum(tr), sum(te))
    rA, rB, rX, nA = Float64.(d.rA), Float64.(d.rB), Float64.(d.rX), Float64.(d.nA)
    t = (rA .+ rX) ./ (sqrt(2) .* (rB .+ rX))                                  # 식 (6.5)
    tau = rX ./ rB .- nA .* (nA .- (rA ./ rB) ./ log.(rA ./ rB))               # 식 (6.6)
    mu = rB ./ rX
    dtau = abs.(tau .- d.tau); k = argmax(dtau)
    @printf("max |t - table| = %.3f; |tau - table|: median %.3f, 99th pct %.3f, max %.2f (at rA/rB=%.3f, %s)\n",
            maximum(abs.(t .- d.t)), median(dtau), quantile(dtau, 0.99), dtau[k], rA[k] / rB[k], d.ABX3[k])
    println("all rA > rB: ", all(rA .> rB))
    pt = Int.((t .> 0.825) .& (t .< 1.059)); pT = Int.(tau .< 4.18)
    @printf("agreement with table predictions: t %.3f, tau %.3f\n", mean(2 .* pt .- 1 .== d.t_pred), mean(2 .* pT .- 1 .== d.tau_pred))
    acc(p, m = trues(length(y))) = mean(p[m] .== y[m])
    println("accuracy            all     train   test")
    @printf("t (0.825<t<1.059)  %.3f   %.3f   %.3f\n", acc(pt), acc(pt, tr), acc(pt, te))
    @printf("tau < 4.18         %.3f   %.3f   %.3f\n", acc(pT), acc(pT, tr), acc(pT, te))
    println("by anion   n   perov%   acc_t   acc_tau")
    for X in ("O", "F", "Cl", "Br", "I")
        m = d.X .== X
        @printf("  %-3s   %4d   %5.1f   %5.3f   %5.3f\n", X, sum(m), 100 * mean(y[m]), acc(pt, m), acc(pT, m))
    end
    sT = cart_fit(reshape(tau[tr], :, 1), y[tr]; max_depth = 1)
    st = cart_fit(reshape(t[tr], :, 1), y[tr]; max_depth = 2)
    accn(node, X, m) = mean(cart_predict(node, X[m, :]) .== y[m])
    @printf("refit on train: tau threshold %.3f -> test acc %.3f ; t depth-2 tree test acc %.3f, thresholds %s\n",
            sT[2], accn(sT, reshape(tau, :, 1), te), accn(st, reshape(t, :, 1), te), round.(sort(cart_thresholds(st)); digits = 3))
    tm = hcat(t, mu); stm = cart_fit(tm[tr, :], y[tr]; max_depth = 3)
    @printf("(t, mu) depth-3 tree: train %.3f test %.3f\n", accn(stm, tm, tr), accn(stm, tm, te))

    # Platt: tau 위의 (사실상 벌점 없는) 로지스틱 회귀, 학습 집합만
    w = logreg_fit(reshape(tau[tr], :, 1), Float64.(y[tr]); C = 1e6)
    a, b = w[1], w[2]; P = sigmoid.(a .* tau .+ b)
    @printf("Platt: P(perovskite) = 1/(1+exp(-(%.3f*tau %+.3f)))  -> P=0.5 at tau=%.3f\n", a, b, -b / a)
    brier = mean((P[te] .- y[te]) .^ 2)
    ll = -mean(y[te] .* log.(P[te]) .+ (1 .- y[te]) .* log.(1 .- P[te]))
    @printf("test: Brier %.4f  log-loss %.4f ; corr with table tau_prob %.3f, mean|diff| %.3f\n",
            brier, ll, cor(P, d.tau_prob), mean(abs.(P .- d.tau_prob)))
    for (lo, hi) in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))
        m = (P .>= lo) .& (P .< hi)
        @printf("  P in [%.1f,%.1f): n=%3d  observed perovskite fraction %.2f\n", lo, min(hi, 1.0), sum(m), mean(y[m]))
    end
end

main()
