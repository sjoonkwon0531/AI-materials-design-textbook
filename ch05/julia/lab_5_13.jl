# Lab 5.13 (Julia) — RPA 보정과 ODT 괄호 찾기
# 파이썬 labs/bcp_rpa.py(Part 1), labs/bcp_odt.py(Part 2), labs/leibler_check.py(Part 0) 대응.
#  Part 0: 연속 Gauss 사슬의 Leibler 평균장 스피노달 (결정론적, 숫자가 그대로 일치해야 함)
#  Part 1: Delta a = 0..4 무질서 A5B5 녹는물(10^3 상자, 300사슬)에서 S_psi(q)와 같은 실행의 블록 형태인자로
#          RPA의 유일한 자유 매개변수 chi_eff를 맞춘다.
#  Part 2: 12x6x6 상자에서 주형 줄무늬 출발과 무질서 출발의 이력으로 ODT를 괄호 친다.
# Part 1·2는 난수 때문에 통계적으로만 일치한다(chi_eff 수 %, Psi는 출력의 표준편차 수준).
using Printf, Random, Statistics, LinearAlgebra
include("dpd_core.jl")
include("bcp_common.jl")

const N = 10; const NA = 5

# ------------------------------------------------------------------ Part 0: Leibler (1980)
debye_g(f, x) = 2 * (f * x + exp(-f * x) - 1) / x^2          # x = q^2 Rg^2
function NGamma(x; f = 0.5)
    saa = debye_g(f, x); sbb = debye_g(1 - f, x); sab = 0.5 * (debye_g(1, x) - saa - sbb)
    return (saa + sbb + 2sab) / (saa * sbb - sab^2)
end

# ------------------------------------------------------------------ Part 1
function melt_run(da, sd; L = 10.0, t_eq = 40.0, t_prod = 240.0)
    box = [L, L, L]; nch = trunc(Int, 3 * L^3 / N)
    Random.seed!(sd)
    x, v, typ, bonds = build_melt(nch, N, 0.5, box); e = [t == 0 ? 1.0 : -1.0 for t in typ]
    A = make_amat(25.0, da); f = zeros(size(x)); pt = zeros(3)
    forces!(f, pt, x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0)
    run!(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, trunc(Int, t_eq / 0.04), 10^6, pt)
    qv = qvectors(box, 4); S = zeros(size(qv, 1)); ns = 0
    qfine = collect(range(0.2, 3.0; length = 57)); ff = zeros(length(qfine), 3); nf = 0; st = zeros(3)
    for blk in 0:trunc(Int, t_prod / 4.0)-1
        run!(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 100, 10^6, pt)
        S .+= sq_psi(x, e, qv); ns += 1
        if blk % 10 == 9
            ff .+= intrachain_sq(x, nch, N, NA, box, qfine); nf += 1
            st .+= chain_stats(x, nch, N, box)
        end
    end
    uq, Sr = radial(qv, S ./ ns)
    return uq, Sr, qfine, ff ./ nf, st ./ nf
end

# ------------------------------------------------------------------ Part 2
const BOX2 = [12.0, 6.0, 6.0]
function qgrid(box; nmax = 4)
    qs = Vector{Float64}[]
    for a in -2nmax:2nmax, b in -nmax:nmax, c in -nmax:nmax
        q = [2pi * a / box[1], 2pi * b / box[2], 2pi * c / box[3]]
        ((a, b, c) > (0, 0, 0) && 0.5 < norm(q) < 2.2) && push!(qs, q)
    end
    return permutedims(hcat(qs...))
end

function odt_run(da, start; sd = 31, t_run = 400.0)
    nch = round(Int, 3 * prod(BOX2) / N); Random.seed!(sd)
    x, v, typ, bonds = build_melt(nch, N, 0.5, BOX2); e = [t == 0 ? 1.0 : -1.0 for t in typ]
    A = make_amat(25.0, da); f = zeros(size(x)); pt = zeros(3)
    forces!(f, pt, x, v, typ, A, 4.5, 3.0, 0.04, BOX2, bonds, 4.0)
    if start == :heat                                        # 주형: 주기 6.0 두 개를 외부장 h=0.5로 t=40
        run!(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, BOX2, bonds, 4.0, 1000, 10^6, pt; h = 0.5, kx = 2pi * 2 / BOX2[1])
    end
    qv = qgrid(BOX2); psi = Float64[]; nblk = trunc(Int, t_run / 8.0)
    for blk in 0:nblk-1
        run!(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, BOX2, bonds, 4.0, 200, 10^6, pt)
        blk >= nblk ÷ 2 && push!(psi, maximum(sq_psi(x, e, qv)) / size(x, 2))
    end
    return mean(psi), std(psi; corrected = false)
end

function main()
    t0 = time()
    xs = fminbound(x -> NGamma(x), 0.5, 10.0; tol = 1e-9)
    println("=== Part 0: Leibler mean-field spinodal, continuous Gaussian chains, f = 1/2 ===")
    @printf("x* = (q* Rg)^2 = %.4f,  q* Rg = %.4f,  (chi N)_s = %.4f\n", xs, sqrt(xs), NGamma(xs) / 2)

    println("=== disordered A5B5 melts: RPA fit of chi_eff (only free parameter) ===")
    das = Float64[]; chis = Float64[]; r0 = nothing
    for da in (0.0, 1.0, 2.0, 3.0, 4.0)
        uq, Sr, qf, ff, st = melt_run(da, 21)
        saa, sbb, sab = (interp1(uq, qf, ff[:, k]) for k in 1:3)
        m = uq .<= 2.6
        function loss(c)
            model = rpa_spsi(saa[m], sbb[m], sab[m], c)
            any(model .<= 0) && return Inf                   # 스피노달 너머: log 정의역 밖
            return sum((log.(model) .- log.(Sr[m])) .^ 2)
        end
        chi = fminbound(loss, -0.5, 0.95); rms = sqrt(loss(chi) / sum(m)); kq = argmax(Sr)
        push!(das, da); push!(chis, chi); da == 0.0 && (r0 = (qf = qf, ff = ff, st = st))
        @printf("  Delta a=%3.1f: chi_eff=%+.4f  (rms log-residual %.3f);  sim peak S=%.2f at q=%.3f;  <Re^2>=%.2f <Rg^2>=%.3f <b^2>=%.3f\n",
                da, chi, rms, Sr[kq], uq[kq], st[1], st[2], st[3])
    end
    p = hcat(das, ones(length(das))) \ chis
    @printf("linear fit: chi_eff = %.3f Delta a %+.3f   (monomer mapping 0.286; polymer mapping 1/3.27 = 0.306)\n", p[1], p[2])
    ff = r0.ff; qf = r0.qf
    gam = (ff[:, 1] .+ ff[:, 2] .+ 2 .* ff[:, 3]) ./ (ff[:, 1] .* ff[:, 2] .- ff[:, 3] .^ 2)
    k = argmin(gam); chis_ = gam[k] / 2; Rg = sqrt(r0.st[2])
    @printf("RPA (measured form factors, Delta a=0): min Gamma at q*=%.2f -> q* Rg = %.2f (continuous Gaussian chain: 1.95);  spinodal chi_s N = %.2f (Leibler: 10.495)\n",
            qf[k], qf[k] * Rg, chis_ * N)
    das_s = (chis_ - p[2]) / p[1]; Re = sqrt(r0.st[1]); Nbar = (3.0 * Re^3 / N)^2
    fh = 10.495 + 41.0 * Nbar^(-1 / 3); gl = fh + 123.0 * Nbar^(-0.56)
    @printf("mean-field spinodal at Delta a = %.2f;  Nbar = (rho Re^3/N)^2 = %.1f;  FH (chi N)_ODT = %.1f,  Glaser et al. fit = %.1f\n",
            das_s, Nbar, fh, gl)

    @printf("box %s, A5B5, %d beads; Psi = max S(q)/n (2nd half of t=400)\n", string(BOX2), round(Int, 3 * prod(BOX2)))
    for da in (6.0, 7.0, 8.0, 9.0, 10.0, 12.0)
        h = odt_run(da, :heat); c = odt_run(da, :cool)
        @printf("  Delta a=%4.1f (chi_GW N=%5.1f): from lamellae Psi=%.4f+/-%.4f   from disorder Psi=%.4f+/-%.4f\n",
                da, 2.86 * da, h[1], h[2], c[1], c[2])
    end
    @printf("wall time %.0f s\n", time() - t0)
end

main()
