# Lab 5.12 (Julia) — DPD 온도 제어, 요동–소산, 그리고 chi–Delta a 사상
# 파이썬 labs/dpd_basics.py 대응 (엔진은 dpd_core.jl).
#  Part 1: rho=3, a=25 유체에서 수정 velocity Verlet의 운동 온도 대 dt, lambda
#  Part 2: sigma^2 = 2 gamma kT를 일부러 어겨 온도가 sigma^2/(2 gamma)로 가는지 확인
#  Part 3: A/B 판 공존 -> 소수 성분 분율 phi -> Flory-Huggins chi (식 5.7.4) -> chi 대 Delta a
# 난수 결과는 통계적으로만 일치한다(각 kT의 통계오차 약 0.005, chi는 출력의 오차막대 참고).
using Printf, Random, Statistics
include("dpd_core.jl")

function simulate(box, typ, amat, gamma, sigma, dt, lam, t_eq, t_run, every, sd)
    Random.seed!(sd); n = length(typ)
    x, v = init_random(n, box); f = zeros(3, n); pt = zeros(3)
    forces!(f, pt, x, v, typ, amat, gamma, sigma, dt, box, NOB, 0.0)
    run!(x, v, f, typ, amat, gamma, sigma, dt, lam, box, NOB, 0.0, trunc(Int, t_eq / dt), 10^6, pt)
    return run!(x, v, f, typ, amat, gamma, sigma, dt, lam, box, NOB, 0.0, trunc(Int, t_run / dt), every, pt)
end

"블록 nb개 평균의 (평균, 표준오차)"
function mean_err(a; nb = 10)
    m = length(a) ÷ nb; b = [mean(@view a[(k-1)*m+1:k*m]) for k in 1:nb]
    return mean(b), std(b) / sqrt(nb)
end

function main()
    T0 = time()
    box = [6.0, 6.0, 6.0]; n = 3 * 216; typ = zeros(Int, n); A = make_amat(25.0, 0.0)
    println("=== Part 1: kinetic temperature vs dt (rho=3, a=25, gamma=4.5, sigma=3, N=648, t=60 after 10) ===")
    P = NaN; ekT_last = NaN
    for lam in (0.5, 0.65)
        row = String[]
        for dt in (0.01, 0.02, 0.04, 0.06, 0.08)
            o = simulate(box, typ, A, 4.5, 3.0, dt, lam, 10.0, 60.0, 5, 3)
            kT, ekT = mean_err(o[:, 1]); push!(row, @sprintf("dt=%.2f: %.4f", dt, kT))
            (dt == 0.01 && lam == 0.5) && (P = mean(o[:, 2:4]))
            ekT_last = dt == 0.01 ? ekT : ekT_last
        end
        @printf("  lambda=%4.2f: %s\n", lam, join(row, "  "))
    end
    @printf("  pressure at dt=0.01: P = %.2f  (k_BT=1, rho=3, a=25)\n", P)
    @printf("  statistical error of each kT ~ %.4f\n", ekT_last)

    println("\n=== Part 2: fluctuation-dissipation check (dt=0.01, lambda=0.5) ===")
    for (gam, sig) in ((4.5, 3.0), (4.5, 4.0), (2.25, 3.0), (9.0, 3.0))
        o = simulate(box, typ, A, gam, sig, 0.01, 0.5, 10.0, 40.0, 5, 5)
        kT, ekT = mean_err(o[:, 1])
        @printf("  gamma=%4.2f sigma=%3.1f: kT = %.4f+/-%.4f   sigma^2/(2 gamma) = %.4f\n", gam, sig, kT, ekT, sig^2 / (2gam))
    end

    println("\n=== Part 3: A/B slab coexistence, box 16x6x6, N=1728, dt=0.04, lambda=0.65 ===")
    Lx = 16.0; boxs = [Lx, 6.0, 6.0]; n = round(Int, 3 * Lx * 36)
    das = Float64[]; chis = Float64[]
    for da in (8.0, 9.0, 10.0, 12.0, 15.0)
        Random.seed!(10)
        x, v = init_random(n, boxs)
        typ = Int.(x[1, :] .>= Lx / 2)                     # A 왼쪽 절반, B 오른쪽 절반
        Am = make_amat(25.0, da); f = zeros(3, n); pt = zeros(3)
        forces!(f, pt, x, v, typ, Am, 4.5, 3.0, 0.04, boxs, NOB, 0.0)
        run!(x, v, f, typ, Am, 4.5, 3.0, 0.04, 0.65, boxs, NOB, 0.0, 2500, 10^6, pt)     # t = 100 평형화
        phis = Float64[]
        for _ in 1:300                                     # 50 step 간격 표본 300개 (t = 600)
            run!(x, v, f, typ, Am, 4.5, 3.0, 0.04, 0.65, boxs, NOB, 0.0, 50, 10^6, pt)
            th = 2pi .* x[1, typ .== 0] ./ Lx              # A 판 중심을 원형 평균으로 찾아 재정렬
            cA = mod(atan(mean(sin.(th)), mean(cos.(th))) * Lx / 2pi, Lx)
            xs = mod.(x[1, :] .- cA .+ Lx / 4, Lx)         # A 판 중심을 Lx/4에
            inA = abs.(xs .- Lx / 4) .< 2.0; inB = abs.(xs .- 3Lx / 4) .< 2.0
            push!(phis, 0.5 * (mean(typ[inA] .== 1) + mean(typ[inB] .== 0)))
        end
        phi, ephi = mean_err(phis)
        chi = log((1 - phi) / phi) / (1 - 2phi)            # 식 (5.7.4)
        dchi = abs(log((1 - phi - ephi) / (phi + ephi)) / (1 - 2 * (phi + ephi)) - chi)
        push!(das, da); push!(chis, chi)
        @printf("  Delta a=%5.1f: minority fraction phi=%.4f+/-%.4f  ->  chi=%.3f+/-%.3f   chi/Delta a=%.3f\n",
                da, phi, ephi, chi, dchi, chi / da)
    end
    slope0 = sum(das .* chis) / sum(das .* das)
    p = hcat(das, ones(length(das))) \ chis
    @printf("  fit through origin: chi = %.3f Delta a   (Groot-Warren: 0.286 Delta a)\n", slope0)
    @printf("  free linear fit:    chi = %.3f Delta a %+.3f\n", p[1], p[2])
    @printf("total wall time %.0f s\n", time() - T0)
end

main()
