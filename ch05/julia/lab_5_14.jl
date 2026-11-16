# Lab 5.14 (Julia) — 무응력 주기와 N^(2/3) 스케일링
# 파이썬 labs/bcp_spacing.py 대응. Delta a = 15의 대칭 A_{N/2}B_{N/2} DPD 녹는물에서 줄무늬 주기 D_eq를
# 무응력 조건 P_N = P_T로 찾는다. N마다 D 세 값에서 <P_xx - (P_yy+P_zz)/2>를 재고 직선의 근을 구한다.
# 난수 때문에 dP와 D_eq는 통계적으로만 일치한다(D_eq는 출력 오차 0.01-0.04 수준, 지수는 약 0.67).
using Printf, Random, Statistics, LinearAlgebra
include("dpd_core.jl")
include("bcp_common.jl")

const DA = 15.0; const LY = 6.0

function dP_at(N, D; sd = 2)
    Lx = 2D; box = [Lx, LY, LY]; nch = round(Int, 3 * Lx * LY * LY / N)
    Random.seed!(sd)
    x, v, typ, bonds = build_melt(nch, N, 0.5, box)
    A = make_amat(25.0, DA); f = zeros(size(x)); pt = zeros(3)
    forces!(f, pt, x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0)
    run!(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1500, 10^6, pt; h = 1.0, kx = 2pi * 2 / Lx)  # 주형 t=60
    run!(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1500, 10^6, pt)                             # 장 끄고 이완
    o = run!(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 8000, 10, pt)                           # [kT Pxx Pyy Pzz]
    dP = o[:, 2] .- 0.5 .* (o[:, 3] .+ o[:, 4])                                                               # P_N - P_T
    m = length(dP) ÷ 10; b = [mean(dP[(k-1)*m+1:k*m]) for k in 1:10]
    e = [t == 0 ? 1.0 : -1.0 for t in typ]
    order = sq_psi(x, e, [2pi * 2 / Lx 0.0 0.0])[1] / size(x, 2)
    return mean(b), std(b) / sqrt(10), order
end

function athermal_Re2(N; sd = 3)
    box = [8.0, 8.0, 8.0]; nch = trunc(Int, 3 * 512 / N)
    Random.seed!(sd)
    x, v, typ, bonds = build_melt(nch, N, 0.5, box)
    A = make_amat(25.0, 0.0); f = zeros(size(x)); pt = zeros(3)
    forces!(f, pt, x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0)
    run!(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1000, 10^6, pt)
    acc = zeros(3)
    for _ in 1:10
        run!(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 200, 10^6, pt)
        acc .+= chain_stats(x, nch, N, box)
    end
    return acc ./ 10
end

"가중 직선 맞춤 (np.polyfit(x, y, 1, w=1/err, cov='unscaled')): 계수 [기울기, 절편]과 공분산"
function wlinfit(x, y, err)
    X = hcat(x, ones(length(x))); Wt = Diagonal(1 ./ err .^ 2)
    cov = inv(X' * Wt * X); return cov * X' * Wt * y, cov
end

function main()
    t0 = time(); Ns = Float64[]; Deqs = Float64[]; as_ = Float64[]
    @printf("Delta a = %s: stress-free lamellar period (box 2D x %s x %s, rho=3)\n", DA, LY, LY)
    for N in (8, 10, 14, 20)
        Dg = 6.45 * (N / 10)^(2 / 3)
        Ds = round.(Dg .* [0.85, 1.0, 1.15]; digits = 2)
        res = [dP_at(N, D) for D in Ds]
        dps = [r[1] for r in res]; errs = [r[2] for r in res]
        p, cov = wlinfit(Ds, dps, errs)
        Deq = -p[2] / p[1]
        g = [p[2] / p[1]^2, -1 / p[1]]; dD = sqrt(g' * cov * g)    # 근의 오차 (선형 전파)
        Re2, Rg2, b2 = athermal_Re2(N); a = sqrt(Re2 / (N - 1))
        push!(Ns, N); push!(Deqs, Deq); push!(as_, a)
        cols = join([@sprintf("D=%5.2f: dP=%+.3f+/-%.3f", D, d, e) for (D, d, e) in zip(Ds, dps, errs)], "  ")
        @printf("  N=%2d: %s  ->  D_eq = %.2f+/-%.2f;  athermal <Re^2>=%.2f, a=%.3f, D_eq/Re0=%.2f\n",
                N, cols, Deq, dD, Re2, a, Deq / sqrt(Re2))
    end
    nu = (hcat(log.(Ns), ones(length(Ns))) \ log.(Deqs))[1]
    @printf("power-law fit D_eq ~ N^%.3f   (strong segregation theory: 2/3; Gaussian coil: 1/2)\n", nu)
    for chi in (0.286 * DA, DA / 3.27)
        pred = 1.098 .* as_ .* chi^(1 / 6) .* Ns .^ (2 / 3)
        @printf("  SST D = 1.098 a chi^(1/6) N^(2/3) with chi=%.2f: %s\n", chi, join([@sprintf("%.2f", d) for d in pred], ", "))
    end
    @printf("wall time %.0f s\n", time() - t0)
end

main()
