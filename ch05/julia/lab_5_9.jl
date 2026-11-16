# Lab 5.9 (Julia) — 미시 KMC로 확산계수를 추정하는 최소 HMM
# 파이썬 labs/56_hmm.py 대응. 1D 확산 du/dt = d/dx(D_eps du/dx), D_eps = A(x) a(x/eps).
#  미세 해상 참값: 2048셀 FV + 후진 Euler.  HMM: 32셀 FV, 경계면 확산계수를 미시 KMC(무작위 장벽
#  보행자)로 추정.  비교: 정확한 균질화 계수(조화평균), 순진한 산술평균.
# 결정론적 부분(미세 해, 균질화·산술평균 거시 해, 유속)은 숫자가 그대로 일치해야 하고
# (lab_5_9_xcheck.py로 대조), 미시 추정과 HMM 행은 난수 때문에 통계적으로만 일치한다.
using Printf, Random, Statistics, LinearAlgebra

const EPS = 1 / 64; const AMP = 0.9
Aslow(x) = 0.5 + x
afast(y) = 1 + AMP * sin(2pi * y)
const HARM = sqrt(1 - AMP^2)                     # <1/a>^{-1}

"후진 Euler FV, n셀, 면 n+1개의 D, 양 끝 Dirichlet. 시각 snaps의 해와 정상해를 Dict로"
function fv_solve(D_face, n, T, dt; u_left = 1.0, u_right = 0.0, snaps = ())
    h = 1.0 / n
    Dl = D_face[1:end-1]; Dr = D_face[2:end]
    Dl[1] *= 2; Dr[end] *= 2                      # 경계 면: 반 셀 거리
    dg = (Dl .+ Dr) ./ h^2; lo = -Dl[2:end] ./ h^2; up = -Dr[1:end-1] ./ h^2
    Lop = Tridiagonal(lo, dg, up)
    F = lu(Tridiagonal(lo .* dt, 1 .+ dt .* dg, up .* dt))   # I + dt L
    bc = zeros(n); bc[1] = Dl[1] / h^2 * u_left; bc[end] = Dr[end] / h^2 * u_right
    u = zeros(n); out = Dict{Any,Vector{Float64}}()
    snap_steps = Dict(round(Int, s / dt) => s for s in snaps)
    for k in 1:round(Int, T / dt)
        u = F \ (u .+ dt .* bc)
        haskey(snap_steps, k) && (out[snap_steps[k]] = copy(u))
    end
    out[:steady] = Lop \ bc
    return out
end

"주기 고리(m자리) 위 무작위 장벽 보행. 결합 k는 자리 k와 k+1을 잇는다. 기록 간격마다 MSD(자리 단위)"
function walk(w_bond, nwalk, nsteps, dtau, seed, nrec)
    Random.seed!(seed)
    m = length(w_bond)
    pr = w_bond .* dtau                                     # 자리 i에서 오른쪽으로 뛸 확률
    pl = [w_bond[mod1(i - 1, m)] * dtau for i in 1:m]       # 왼쪽
    every = nsteps ÷ nrec; msd = zeros(nrec)
    for _ in 1:nwalk
        i = rand(1:m); pos = 0                              # 균일 출발 = 정상분포
        for s in 1:nsteps
            u = rand()
            if u < pr[i]
                pos += 1; i = i == m ? 1 : i + 1
            elseif u < pr[i] + pl[i]
                pos -= 1; i = i == 1 ? m : i - 1
            end
            s % every == 0 && (msd[s÷every] += pos * pos)
        end
    end
    return msd ./ nwalk
end

"거시점 x0의 유효 D 추정: 미시 셀 = 한 주기 eps, m개 격자점, 느린 변수는 A(x0)로 고정"
function micro_D(x0; m = 16, nwalk = 4000, nsteps = 8000, seed = 0, nrec = 80)
    h = EPS / m; y = ((0:m-1) .+ 0.5) ./ m
    D_bond = Aslow(x0) .* afast.(y); w = D_bond ./ h^2
    dtau = 0.5 / maximum(w)
    msd = walk(w, nwalk, nsteps, dtau, seed, nrec) .* h^2
    t = dtau .* ((1:nrec) .* (nsteps ÷ nrec))
    D_t = msd ./ (2 .* t)
    half = nrec ÷ 2                                         # 뒤쪽 절반(Fick 영역)에 직선 맞춤
    X = hcat(t[half+1:end], ones(nrec - half)); slope = (X \ msd[half+1:end])[1]
    return slope / 2, t, D_t, t[end] * mean(D_bond) / EPS^2
end

trapz(x, y) = sum((x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1])) / 2

function main()
    t0 = time(); x0 = 0.5
    D0, t_m, D_t, tcross = micro_D(x0; seed = 1)
    @printf("[1] micro cell at x0=%s: A=%.3f, arithmetic A<a>=%.4f, harmonic A<1/a>^-1=%.4f\n", x0, Aslow(x0), Aslow(x0), Aslow(x0) * HARM)
    @printf("    D(t)=MSD/2t at first record = %.4f, at last record = %.4f\n", D_t[1], D_t[end])
    yb = ((0:15) .+ 0.5) ./ 16
    @printf("    lattice-exact harmonic mean (16 bonds) = %.4f\n", Aslow(x0) / mean(1 ./ afast.(yb)))
    @printf("    slope estimate (second half) = %.4f;  micro run length = %.1f x eps^2/<D>\n", D0, tcross)
    reps = [micro_D(x0; seed = s)[1] for s in 10:19]
    @printf("    10 independent micro runs: D = %.4f +/- %.4f (std), exact %.4f\n", mean(reps), std(reps), Aslow(x0) * HARM)

    M = 32; xf_macro = collect(range(0, 1; length = M + 1))
    D_hmm = [micro_D(x; seed = 100 + k - 1)[1] for (k, x) in enumerate(xf_macro)]
    D_hom = Aslow.(xf_macro) .* HARM; D_ari = Aslow.(xf_macro)
    rel = (D_hmm .- D_hom) ./ D_hom
    @printf("\n[2] HMM interface diffusivities (M=%d): mean |rel. error| vs homogenized = %.3f, max = %.3f\n",
            M, mean(abs.(rel)), maximum(abs.(rel)))

    T, dt = 0.1, 1e-4; snaps = (0.01, 0.1)
    n_f = 2048; xf_fine = collect(range(0, 1; length = n_f + 1))
    t1 = time(); fine = fv_solve(Aslow.(xf_fine) .* afast.(xf_fine ./ EPS), n_f, T, dt; snaps = snaps); t_fine = time() - t1
    res = Dict(name => fv_solve(D, M, T, dt; snaps = snaps)
               for (name, D) in (("HMM", D_hmm), ("homogenized", D_hom), ("arithmetic", D_ari)))
    coarse(u) = vec(mean(reshape(u, :, M); dims = 1))      # 거시 셀마다 미세 해 평균 (u.reshape(M,-1).mean(1))
    println("\n[3] macro solution error vs fine-scale truth (cell averages, max norm)")
    println("   method         t=0.01     t=0.1      steady")
    for name in ("HMM", "homogenized", "arithmetic")
        e = [maximum(abs.(res[name][k] .- coarse(fine[k]))) for k in (0.01, 0.1, :steady)]
        @printf("   %-12s  %.4f    %.4f    %.4f\n", name, e[1], e[2], e[3])
    end
    xs = collect(range(0, 1; length = 200001))
    J_fine = 1 / trapz(xs, 1 ./ (Aslow.(xs) .* afast.(xs ./ EPS)))   # 정확: J = 1 / int(1/D_eps)
    J_hom = 1 / trapz(xs, 1 ./ (Aslow.(xs) .* HARM)); J_ari = 1 / trapz(xs, 1 ./ Aslow.(xs))
    @printf("\n[4] steady flux: fine (exact quadrature) = %.4f, homogenized = %.4f, arithmetic = %.4f\n", J_fine, J_hom, J_ari)
    for (name, D) in (("HMM", D_hmm), ("homogenized", D_hom), ("arithmetic", D_ari))
        us = res[name][:steady]
        @printf("    macro steady flux (%s) = %.4f\n", name, D[1] * (1.0 - us[1]) / (0.5 / M))
    end
    @printf("\n[5] cost: fine FV %d cells x %d steps; HMM macro %d cells x %d steps + micro %.2e walker-steps (once, coefficient is time-independent)\n",
            n_f, round(Int, T / dt), M, round(Int, T / dt), (M + 1) * 4000 * 8000)
    @printf("    fine-solver wall time %.2f s\n", t_fine)
    @printf("\nruntime %.1f s\n", time() - t0)
end

main()
