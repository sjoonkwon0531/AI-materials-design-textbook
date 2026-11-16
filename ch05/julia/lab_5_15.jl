# Lab 5.15 (Julia) — 지문, 화학 에피택시, 밀도 증대
# 파이썬 labs/bcp_film.py 대응. Delta a = 15의 A5B5 DPD 박막(4 L0 x 4 L0 x 3, L0 = 6.41, 5920 비드)을
# 같은 무질서 출발에서 (i) 안내 없이, (ii) 밀한 패턴 U = -h e cos(2 pi x / L0), h = 0.2,
# (iii) 성긴 패턴(간격 2 L0, 폭 0.8의 A 유인 Gauss 줄무늬, h = 1.0; 2배 밀도 증대)으로 t = 400 조립.
# 질서도: max S(q)/n과 P2 = <cos 2 theta>_S (|q| in [0.6, 1.4]).
# 난수 때문에 통계적으로만 일치한다. 특히 안내 없는 지문 조직의 P2 부호와 주기는 표본마다 다르지만,
# 안내된 두 경우의 S/n ~ 0.37-0.39, P2 ~ +0.96-0.98, 주기 6.41은 재현되어야 한다.
using Printf, Random, Statistics, LinearAlgebra
include("dpd_core.jl")
include("bcp_common.jl")

const L0 = 6.41; const NCH_N = 10; const DAF = 15.0

function film(case; t_run = 400.0, sd = 5)
    box = [4L0, 4L0, 3.0]; nch = round(Int, 3 * prod(box) / NCH_N)
    Random.seed!(sd)
    x, v, typ, bonds = build_melt(nch, NCH_N, 0.5, box); e = [t == 0 ? 1.0 : -1.0 for t in typ]
    A = make_amat(25.0, DAF); f = zeros(size(x)); pt = zeros(3)
    forces!(f, pt, x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0)
    h, k, mode, w = Dict(:unguided => (0.0, 1.0, 0, 1.0), :dense => (0.2, 2pi / L0, 0, 1.0),
                         :sparse => (1.0, 2pi / (2L0), 1, 0.8))[case]
    qs = [[2pi * a / box[1], 2pi * b / box[2], 0.0] for a in -8:8 for b in 0:8 if (b > 0 || a > 0)]
    qv = permutedims(hcat(qs...)); qm = sqrt.(qv[:, 1] .^ 2 .+ qv[:, 2] .^ 2); th = atan.(qv[:, 2], qv[:, 1])
    sel = (qm .> 0.6) .& (qm .< 1.4); hist = zeros(0, 4)
    for blk in 1:trunc(Int, t_run / 50)
        run!(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1250, 10^6, pt; h = h, kx = k, mode = mode, w = w)
        s = sq_psi(x, e, qv); j = argmax(s)
        P2 = sum(s[sel] .* cos.(2 .* th[sel])) / sum(s[sel])
        hist = vcat(hist, [blk * 50 s[j] / size(x, 2) qm[j] P2])
    end
    return x, typ, e, hist
end

function main()
    t0 = time()
    @printf("film 4L0 x 4L0 x 3, L0=%s, A5B5, Delta a=%s, t=400 (10^4 steps, dt=0.04)\n", L0, DAF)
    for case in (:unguided, :dense, :sparse)
        x, typ, e, hist = film(case)
        t, S, q, P2 = hist[end, :]
        @printf("  %-8s: t=%.0f  max S(q)/n=%.3f at |q|=%.3f (period %.2f)  P2=%+.2f   [t=100: S/n=%.3f, P2=%+.2f]\n",
                string(case), t, S, q, 2pi / q, P2, hist[2, 2], hist[2, 4])
        qq = [2pi/(2L0) 0 0; 2pi/L0 0 0; 3*2pi/L0 0 0]
        sx = sq_psi(x, e, qq) ./ size(x, 2)
        @printf("            S/n along x at q = 2pi/(2L0), 2pi/L0, 3(2pi/L0): %.4f, %.4f, %.4f\n", sx[1], sx[2], sx[3])
    end
    @printf("wall time %.0f s\n", time() - t0)
end

main()
