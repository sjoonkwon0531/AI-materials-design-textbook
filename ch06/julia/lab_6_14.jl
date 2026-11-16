# Lab 6.14 (Julia) — 합성 인벤토리의 Monte Carlo LCA: 불확도, 민감도, 비교
# 파이썬 labs/66_mclca.py 대응. 모든 인벤토리 양과 배경 배출계수는 교육용 합성 값이다(인용 금지).
# 실제 값은 특성화 계수(IPCC AR6 GWP100: CO2 1, 화석 CH4 29.8, N2O 273; 비교용 AR5 28, 265)뿐이다.
# 결정론적 LCIA(중앙값)와 AR5 비교, 곱셈 항의 GSD 규칙(이론값)은 그대로 일치해야 하고,
# Monte Carlo 결과는 Julia RNG라서 통계적으로 일치한다(N = 10^5에서 평균의 표준오차 ~0.02).
using Printf, Random, Statistics, StatsBase

const GWP = Dict("AR6" => Dict("CO2" => 1.0, "CH4" => 29.8, "N2O" => 273.0),
                 "AR5" => Dict("CO2" => 1.0, "CH4" => 28.0, "N2O" => 265.0))
# 배경 배출계수 (중앙값, GSD) — 합성
const EF = Dict("elec_CO2" => (0.45, 1.25), "elec_CH4" => (1.0e-3, 2.0), "elec_N2O" => (1.0e-5, 2.0),
                "gas_CO2" => (0.056, 1.05), "gas_CH4" => (3.0e-4, 2.0),
                "P1" => (3.0, 1.5), "P2" => (9.0, 1.6), "solv" => (1.8, 1.4))
const EF_KEYS = ["elec_CO2", "elec_CH4", "elec_N2O", "gas_CO2", "gas_CH4", "P1", "P2", "solv"]
# 전경 인벤토리 (기능 단위 1 kg당; 중앙값, GSD) — 합성
const INV = Dict(
    "A" => Dict("P1" => (1.10, 1.05), "P2" => (0.45, 1.05), "elec" => (30.0, 1.3), "gas" => (40.0, 1.2),
                "proc_CO2" => (0.30, 1.5), "solv" => (0.0, 1.0)),
    "B" => Dict("P1" => (1.05, 1.05), "P2" => (0.45, 1.05), "elec" => (12.0, 1.3), "gas" => (70.0, 1.2),
                "proc_CO2" => (0.05, 1.5), "solv" => (0.60, 1.5)))
const INV_KEYS = ["P1", "P2", "elec", "gas", "proc_CO2", "solv"]
const CONTRIB = ["P1", "P2", "electricity", "heat", "process CO2", "solvent"]

ln_sample(rng, med, gsd, n) = gsd > 1 ? med .* exp.(log(gsd) .* randn(rng, n)) : fill(med, n)
draw_ef(rng, n; ef = EF) = Dict(k => ln_sample(rng, ef[k]..., n) for k in EF_KEYS)        # 키 순서 고정
draw_inv(rng, r, n) = Dict(k => ln_sample(rng, INV[r][k]..., n) for k in INV_KEYS)

"기여도(kg CO2e / 기능 단위). 입력은 스칼라 또는 같은 길이의 벡터"
function gwp(ef, inv, cf)
    e_elec = inv["elec"] .* (ef["elec_CO2"] .* cf["CO2"] .+ ef["elec_CH4"] .* cf["CH4"] .+ ef["elec_N2O"] .* cf["N2O"])
    e_gas = inv["gas"] .* (ef["gas_CO2"] .* cf["CO2"] .+ ef["gas_CH4"] .* cf["CH4"])
    return Dict("P1" => inv["P1"] .* ef["P1"], "P2" => inv["P2"] .* ef["P2"], "electricity" => e_elec,
                "heat" => e_gas, "process CO2" => inv["proc_CO2"] .* cf["CO2"], "solvent" => inv["solv"] .* ef["solv"])
end
total(c) = sum(c[k] for k in CONTRIB)

function main()
    t0 = time(); cf = GWP["AR6"]
    det_ef = Dict(k => v[1] for (k, v) in EF)
    det = Dict(r => gwp(det_ef, Dict(k => v[1] for (k, v) in INV[r]), cf) for r in ("A", "B"))
    println("=== Deterministic LCIA (medians), kg CO2e per kg X, GWP100 (AR6) ===")
    for r in ("A", "B")
        tot = total(det[r])
        println("route $r: total ", @sprintf("%.2f", tot), " | ",
                join([@sprintf("%s %.2f (%.0f%%)", k, det[r][k], 100 * det[r][k] / tot) for k in CONTRIB if det[r][k] > 0], ", "))
    end
    a5 = Dict(r => total(gwp(det_ef, Dict(k => v[1] for (k, v) in INV[r]), GWP["AR5"])) for r in ("A", "B"))
    @printf("with AR5 factors: A %.2f, B %.2f\n", a5["A"], a5["B"])

    N = 100_000; rng = MersenneTwister(42)
    ef = draw_ef(rng, N); inv = Dict(r => draw_inv(rng, r, N) for r in ("A", "B"))
    tot = Dict(r => total(gwp(ef, inv[r], cf)) for r in ("A", "B"))
    println("=== Monte Carlo, N=$N, seed 42 (Julia MersenneTwister) ===")
    for r in ("A", "B")
        t = tot[r]; q = quantile(t, [0.025, 0.5, 0.975])
        @printf("route %s: mean %.2f  median %.2f  95%% interval [%.2f, %.2f]  sd %.2f  (mean/deterministic = %.3f)\n",
                r, mean(t), q[2], q[1], q[3], std(t; corrected = false), mean(t) / total(det[r]))
    end
    D = tot["A"] - tot["B"]
    @printf("A-B, shared background (correlated): mean %.2f  sd %.2f  P(A<B) = %.3f\n", mean(D), std(D; corrected = false), mean(D .< 0))
    totB_ind = total(gwp(draw_ef(rng, N), inv["B"], cf))
    D2 = tot["A"] - totB_ind
    @printf("A-B, independent background (wrong):  mean %.2f  sd %.2f  P(A<B) = %.3f\n", mean(D2), std(D2; corrected = false), mean(D2 .< 0))

    println("=== Sensitivity of route A total: normalised squared rank correlation ===")
    inputs = vcat([("EF $k", ef[k]) for k in EF_KEYS if k != "solv"], [("inv $k", inv["A"][k]) for k in INV_KEYS if k != "solv"])
    rs = [(k, corspearman(v, tot["A"])) for (k, v) in inputs]
    S = sum(r^2 for (_, r) in rs)
    for (k, r) in sort(rs; by = z -> -z[2]^2)[1:6]
        @printf("  %-12s rank corr %+.3f  share %5.1f%%\n", k, r, 100 * r^2 / S)
    end

    e_el = inv["A"]["elec"] .* ef["elec_CO2"]
    gsd_mc = exp(std(log.(e_el); corrected = false)); gsd_th = exp(sqrt(log(1.3)^2 + log(1.25)^2))
    @printf("electricity CO2 term of A: GSD (MC) = %.4f, sqrt(ln^2 1.3 + ln^2 1.25) rule = %.4f\n", gsd_mc, gsd_th)
    med_th = 30 * 0.45; mean_th = med_th * exp(0.5 * log(gsd_th)^2)
    @printf("  median %.3f (theory %.3f),  mean %.3f (theory %.3f)\n", median(e_el), med_th, mean(e_el), mean_th)

    rng = MersenneTwister(7)
    EF_low = copy(EF); EF_low["elec_CO2"] = (0.05, 1.25); EF_low["elec_CH4"] = (1.0e-3 / 9, 2.0); EF_low["elec_N2O"] = (1.0e-5 / 9, 2.0)
    efl = draw_ef(rng, N; ef = EF_low)
    totl = Dict(r => total(gwp(efl, draw_inv(rng, r, N), cf)) for r in ("A", "B"))
    @printf("=== low-carbon grid scenario: A median %.2f, B median %.2f, P(A<B) = %.3f\n",
            median(totl["A"]), median(totl["B"]), mean(totl["A"] .< totl["B"]))

    println("=== MC convergence (route A mean) ===")
    for n in (100, 1000, 10000, 100000)
        means = [mean(total(gwp(draw_ef(MersenneTwister(1000 + s), n), draw_inv(MersenneTwister(2000 + s), "A", n), cf))) for s in 0:19]
        @printf("  N=%6d: sd of mean over 20 seeds = %.4f   sd/sqrt(N) estimate = %.4f\n", n, std(means), std(tot["A"]; corrected = false) / sqrt(n))
    end
    @printf("time %.1f s\n", time() - t0)
end

main()
