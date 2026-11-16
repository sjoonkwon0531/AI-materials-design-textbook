# Lab 5.3 (Julia) — 유한 크기 스케일링으로 2D·3D Ising의 T_c 뽑아내기
# 파이썬 labs/52_fss.py(2D), labs/52_fss3d.py(3D), labs/reweight.py 대응.
# L마다 한 온도의 Wolff 시뮬레이션 하나 + 단일 히스토그램 재가중(Ferrenberg-Swendsen) + 20블록 jackknife.
# 난수가 numpy와 다르므로 교차 온도·U4 등은 jackknife 오차 안에서 통계적으로 일치한다.
# 실행 시간 절약이 필요하면 RUN3D = false로 2D 부분만 돌린다.
using Printf, Random, Statistics, LinearAlgebra, FFTW
include("isingkern.jl")
const RUN3D = true

# ------------------------------------------------------------------ 재가중 (reweight.py)
"beta0에서 뽑은 (E, M) 시계열을 각 beta로 재가중. log-sum-exp로 넘침 방지."
function rw_moments(E, M, N, beta0, betas)
    m = abs.(M) ./ N; nb = length(betas)
    am = zeros(nb); m2 = zeros(nb); m4 = zeros(nb); e = zeros(nb)
    for (k, b) in enumerate(betas)
        lw = -(b - beta0) .* E
        w = exp.(lw .- maximum(lw)); Z = sum(w)
        am[k] = sum(w .* m) / Z; m2[k] = sum(w .* m .^ 2) / Z
        m4[k] = sum(w .* m .^ 4) / Z; e[k] = sum(w .* E) / Z / N
    end
    return (am = am, m2 = m2, m4 = m4, e = e)
end

function observables(mom, betas, N)
    U4 = 1.0 .- mom.m4 ./ (3.0 .* mom.m2 .^ 2)                 # Binder 누적량
    chi = betas .* N .* mom.m2
    chip = betas .* N .* (mom.m2 .- mom.am .^ 2)               # chi'
    return U4, chi, chip
end

"연속 블록 nblk개 jackknife. func(mom) -> 배열. (전체값, 오차, 복제값 행렬 nblk x len)"
function jackknife(E, M, N, beta0, betas, func; nblk = 20)
    n = length(E) ÷ nblk * nblk; bl = n ÷ nblk
    E = E[1:n]; M = M[1:n]
    full = func(rw_moments(E, M, N, beta0, betas))
    reps = zeros(nblk, length(full))
    for k in 1:nblk
        keep = vcat(1:(k-1)*bl, k*bl+1:n)                     # k번째 블록을 뺀 나머지
        reps[k, :] = func(rw_moments(E[keep], M[keep], N, beta0, betas))
    end
    err = vec(sqrt.((nblk - 1) / nblk .* sum((reps .- mean(reps; dims = 1)) .^ 2; dims = 1)))
    return full, err, reps
end

"선형 보간 (np.interp, xs 증가)"
function interp1(x, xs, ys)
    k = clamp(searchsortedlast(xs, x), 1, length(xs) - 1)
    return ys[k] + (ys[k+1] - ys[k]) * (x - xs[k]) / (xs[k+1] - xs[k])
end

"ua - ub의 부호 변화 중 Tref에 가장 가까운 것을 선형 보간"
function crossing(T, ua, ub, Tref)
    d = ua .- ub
    idx = [k for k in 1:length(d)-1 if sign(d[k]) != sign(d[k+1])]
    isempty(idx) && return NaN
    k = idx[argmin(abs.(T[idx] .- Tref))]
    return T[k] - d[k] * (T[k+1] - T[k]) / (d[k+1] - d[k])
end

jkerr(cj) = sqrt((length(cj) - 1) / length(cj) * sum((cj .- mean(cj)) .^ 2))

# ------------------------------------------------------------------ 2D
function fss2d()
    t0 = time(); bc = 0.5 * log(1 + sqrt(2)); Tc = 1 / bc
    runs = [(8, 200000), (16, 150000), (32, 80000), (64, 30000)]
    T = collect(range(2.0, 2.6; length = 301)); betas = 1 ./ T
    data = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    @printf("=== 2D Ising, Wolff (measure once per sweep-equivalent), single run per L at T0 = T_c = %.6f ===\n", Tc)
    for (L, n) in runs
        t = time()
        E, M = run_wolff2d_sweeps(L, bc, 5000, n, 300 + L); data[L] = (E, M); N = L * L
        u0, du0, _ = jackknife(E, M, N, bc, [bc], mom -> observables(mom, [bc], N)[1])
        @printf("  L=%3d: %6d sweeps, U4(T_c) = %.4f +/- %.4f   (%.0f s)\n", L, n, u0[1], du0[1], time() - t)
    end
    # 측정 시점의 함정 (★Why 5.4): 같은 L=16, 같은 측정 수
    Eb, Mb = run_wolff2d_sweeps(16, bc, 5000, 150000, 316, true)
    ub, dub, _ = jackknife(Eb, Mb, 256, bc, [bc], mom -> observables(mom, [bc], 256)[1])
    @printf("  pitfall, L=16 measured 'after >= N flipped spins' (state-dependent stopping): U4(T_c) = %.4f +/- %.4f  (biased)\n",
            ub[1], dub[1])

    curves = Dict{Int,Vector{Float64}}(); reps = Dict{Int,Matrix{Float64}}()
    for (L, _) in runs
        E, M = data[L]; N = L * L
        curves[L], _, reps[L] = jackknife(E, M, N, bc, betas, mom -> observables(mom, betas, N)[1])
    end
    println("=== Binder crossings of reweighted U4(T) (jackknife errors) ===")
    Ls = [L for (L, _) in runs]
    for (a, b) in zip(Ls[1:end-1], Ls[2:end])
        c = crossing(T, curves[a], curves[b], Tc)
        cj = [crossing(T, reps[a][k, :], reps[b][k, :], Tc) for k in 1:size(reps[a], 1)]
        ustar = interp1(c, T, curves[b])
        @printf("  (%2d,%3d): T* = %.5f +/- %.5f   (T*-T_c)/T_c = %+.1e   U* = %.4f\n", a, b, c, jkerr(cj), (c - Tc) / Tc, ustar)
    end
    @printf("  exact T_c = %.5f\n", Tc)

    println("=== check: reweighted (from T_c run) vs direct Wolff run, L=32 ===")
    L = 32; N = L * L; E, M = data[L]; sigE = std(E; corrected = false)      # np.std (ddof=0)
    @printf("  energy std at T_c: sigma_E = %.1f  -> window |dbeta| ~ 1/sigma_E = %.4f  (|dT| ~ %.3f)\n", sigE, 1 / sigE, Tc^2 / sigE)
    for Td in (2.23, 2.31, 2.60)
        bd = 1 / Td; f = mom -> observables(mom, [bd], N)[1]
        rw = f(rw_moments(E, M, N, bc, [bd]))[1]
        Ed, Md = run_wolff2d_sweeps(L, bd, 5000, 40000, 900 + trunc(Int, 100 * Td))
        dr = f(rw_moments(Ed, Md, N, bd, [bd]))[1]
        _, edr, _ = jackknife(Ed, Md, N, bd, [bd], f)
        _, erw, _ = jackknife(E, M, N, bc, [bd], f)
        @printf("  T=%.2f (|dbeta|*sigma_E = %4.1f): reweighted U4 = %.4f +/- %.4f   direct U4 = %.4f +/- %.4f\n",
                Td, abs(bd - bc) * sigE, rw, erw[1], dr, edr[1])
    end

    println("=== data collapse: chi' L^(-gamma/nu) vs t L^(1/nu), gamma = 7/4, nu = 1 ===")
    chimax = Float64[]
    for L in Ls
        E, M = data[L]; N = L * L
        U4, chi, chip = observables(rw_moments(E, M, N, bc, betas), betas, N)
        x = (T .- Tc) ./ Tc .* L
        k = argmax(chip); push!(chimax, chip[k])
        @printf("  L=%3d: chi'_max = %8.3f at T = %.4f  (x = %.2f),  chi'_max L^-7/4 = %.4f\n",
                L, chip[k], T[k], x[k], chip[k] * L^-1.75)
    end
    @printf("  fit chi'_max ~ L^%.3f  (gamma/nu = 1.75)\n", linfit(log.(Ls), log.(chimax))[1])
    @printf("total %.0f s\n", time() - t0)
end

# ------------------------------------------------------------------ 3D
function fss3d()
    t0 = time(); b0 = 0.2216546
    runs = [(6, 60000), (8, 60000), (12, 40000), (16, 30000), (24, 15000), (32, 8000)]
    T = collect(range(4.44, 4.58; length = 281)); betas = 1 ./ T
    data = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    @printf("=== 3D Ising (simple cubic), Wolff, one run per L at beta0 = %s (T0 = %.4f) ===\n", b0, 1 / b0)
    for (L, n) in runs
        t = time()
        E, M = run_wolff3d_sweeps(L, b0, 3000, n, 500 + L); data[L] = (E, M); N = L^3
        u, du, _ = jackknife(E, M, N, b0, [b0], mom -> observables(mom, [b0], N)[1])
        sE = std(E; corrected = false)
        @printf("  L=%3d: %6d sweeps, U4(T0) = %.4f +/- %.4f, sigma_E*|dbeta| at T=4.44/4.58: %.1f/%.1f   (%.0f s)\n",
                L, n, u[1], du[1], sE * abs(1 / 4.44 - b0), sE * abs(1 / 4.58 - b0), time() - t)
    end
    curves = Dict{Int,Vector{Float64}}(); reps = Dict{Int,Matrix{Float64}}()
    for (L, _) in runs
        E, M = data[L]; N = L^3
        curves[L], _, reps[L] = jackknife(E, M, N, b0, betas, mom -> observables(mom, betas, N)[1])
    end
    pairs = [(6, 12), (8, 16), (12, 24), (16, 32)]
    println("=== Binder crossings T*(L, 2L) ===")
    La = Float64[]; Ts = Float64[]; eT = Float64[]
    for (a, b) in pairs
        c = crossing(T, curves[a], curves[b], 1 / b0)
        cj = [crossing(T, reps[a][k, :], reps[b][k, :], 1 / b0) for k in 1:size(reps[a], 1)]
        push!(La, a); push!(Ts, c); push!(eT, jkerr(cj))
        @printf("  (%2d,%2d): T* = %.4f +/- %.4f   U* = %.4f\n", a, b, c, eT[end], interp1(c, T, curves[b]))
    end
    # 스케일링 보정 외삽: 지수 omega + 1/nu 고정, 가중 최소제곱
    w_exp = 0.83 + 1 / 0.630
    X = hcat(ones(length(La)), La .^ (-w_exp)); W = Diagonal(1 ./ eT .^ 2)
    cov = inv(X' * W * X); coef = cov * X' * W * Ts
    chi2 = sum((Ts .- X * coef) .^ 2 ./ eT .^ 2)
    @printf("=== weighted fit T*(L) = T_c + a L^-%.2f over L = %s ===\n", w_exp, string(Int.(La)))
    @printf("  T_c = %.4f +/- %.4f,  a = %.3f,  chi2/dof = %.2f\n", coef[1], sqrt(cov[1, 1]), coef[2], chi2 / (length(La) - 2))
    @printf("  naive (largest pair only): T* = %.4f;  reference T_c = 1/0.221654626 = %.4f\n", Ts[end], 1 / 0.221654626)
    @printf("total %.0f s\n", time() - t0)
end

fss2d()
RUN3D && fss3d()
