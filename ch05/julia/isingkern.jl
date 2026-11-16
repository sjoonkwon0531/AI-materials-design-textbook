# isingkern.jl — Lab 5.2·5.3 공용 Ising 커널 (2D·3D, J = 1, k_B = 1). 파이썬 labs/isingkern.py 대응.
# 1-based 인덱스: 주기 이웃은 mod1(i ± 1, L). 스핀 배열은 Matrix{Int}/Array{Int,3}.
using Random, Statistics, FFTW

# ------------------------------------------------------------------ 2D
function energy2d(s)
    L = size(s, 1); e = 0
    @inbounds for j in 1:L, i in 1:L
        e -= s[i, j] * (s[mod1(i + 1, L), j] + s[i, mod1(j + 1, L)])
    end
    return e
end

"L^2번의 무작위 자리 Metropolis 시도 (Ch.1 알고리즘). (E, M)을 갱신해 돌려준다."
function metro_sweep2d!(s, beta, E, M)
    L = size(s, 1); w4 = exp(-4.0 * beta); w8 = exp(-8.0 * beta)
    @inbounds for _ in 1:L*L
        i = rand(1:L); j = rand(1:L)
        nb = s[mod1(i + 1, L), j] + s[mod1(i - 1, L), j] + s[i, mod1(j + 1, L)] + s[i, mod1(j - 1, L)]
        dE = 2 * s[i, j] * nb
        if dE <= 0 || (dE == 4 && rand() < w4) || (dE == 8 && rand() < w8)
            s[i, j] = -s[i, j]; E += dE; M += 2 * s[i, j]
        end
    end
    return E, M
end

const NB2 = ((1, 0), (-1, 0), (0, 1), (0, -1))

"Wolff 클러스터 하나(키우면서 뒤집기). 클러스터 크기를 돌려준다."
function wolff2d!(s, padd, si, sj)
    L = size(s, 1)
    i = rand(1:L); j = rand(1:L); s0 = s[i, j]; s[i, j] = -s0
    si[1] = i; sj[1] = j; top = 1; size_ = 1
    @inbounds while top > 0
        a = si[top]; b = sj[top]; top -= 1
        for (di, dj) in NB2
            x = mod1(a + di, L); y = mod1(b + dj, L)
            if s[x, y] == s0 && rand() < padd
                s[x, y] = -s0; top += 1; si[top] = x; sj[top] = y; size_ += 1
            end
        end
    end
    return size_
end

"에너지 변화까지 돌려주는 Wolff (소속 표시 배열 mark, 클러스터 목록 ci/cj 사용)"
function wolff2d_dE!(s, padd, si, sj, mark, stamp, ci, cj)
    L = size(s, 1)
    i = rand(1:L); j = rand(1:L); s0 = s[i, j]; s[i, j] = -s0; mark[i, j] = stamp
    si[1] = i; sj[1] = j; top = 1; size_ = 1; ci[1] = i; cj[1] = j
    @inbounds while top > 0
        a = si[top]; b = sj[top]; top -= 1
        for (di, dj) in NB2
            x = mod1(a + di, L); y = mod1(b + dj, L)
            if s[x, y] == s0 && rand() < padd
                s[x, y] = -s0; mark[x, y] = stamp
                top += 1; si[top] = x; sj[top] = y
                size_ += 1; ci[size_] = x; cj[size_] = y
            end
        end
    end
    dE = 0                                    # 클러스터 경계 결합만 부호가 바뀐다
    @inbounds for c in 1:size_
        a = ci[c]; b = cj[c]
        for (di, dj) in NB2
            x = mod1(a + di, L); y = mod1(b + dj, L)
            mark[x, y] != stamp && (dE += 2 * s0 * s[x, y])
        end
    end
    return size_, dE, s0
end

function run_metro2d(L, beta, n_eq, n_meas, seed)
    Random.seed!(seed)
    s = ones(Int, L, L)
    for i in 1:L, j in 1:L
        rand() < 0.5 && (s[i, j] = -1)
    end
    E = energy2d(s); M = sum(s)
    for _ in 1:n_eq
        E, M = metro_sweep2d!(s, beta, E, M)
    end
    Es = zeros(n_meas); Ms = zeros(n_meas)
    for t in 1:n_meas
        E, M = metro_sweep2d!(s, beta, E, M); Es[t] = E; Ms[t] = M
    end
    return Es, Ms
end

"클러스터마다 측정. E, M은 증분으로 추적. (E, M, 클러스터 크기) 시계열"
function run_wolff2d(L, beta, n_eq, n_meas, seed)
    Random.seed!(seed)
    s = ones(Int, L, L); padd = 1.0 - exp(-2.0 * beta)
    si = zeros(Int, L * L); sj = zeros(Int, L * L); ci = zeros(Int, L * L); cj = zeros(Int, L * L)
    mark = zeros(Int, L, L); stamp = 0
    E = energy2d(s); M = sum(s)
    for _ in 1:n_eq
        stamp += 1
        c, dE, s0 = wolff2d_dE!(s, padd, si, sj, mark, stamp, ci, cj)
        E += dE; M -= 2 * s0 * c
    end
    Es = zeros(n_meas); Ms = zeros(n_meas); Cs = zeros(n_meas)
    for t in 1:n_meas
        stamp += 1
        c, dE, s0 = wolff2d_dE!(s, padd, si, sj, mark, stamp, ci, cj)
        E += dE; M -= 2 * s0 * c
        Cs[t] = c; Es[t] = E; Ms[t] = M
    end
    return Es, Ms, Cs
end

"""sweep 환산 1회마다 측정. 올바른 판(biased=false): 클러스터 간격 ncl = round(N/<C>)를
평형화 구간에서 한 번 정하고 고정. biased=true: 'N개 이상 뒤집힐 때까지' (상태 의존 정지 규칙, ★Why 5.4)."""
function run_wolff2d_sweeps(L, beta, n_eq_cl, n_meas, seed, biased = false)
    Random.seed!(seed)
    N = L * L; s = ones(Int, L, L); padd = 1.0 - exp(-2.0 * beta)
    si = zeros(Int, N); sj = zeros(Int, N)
    csum = 0.0
    for _ in 1:n_eq_cl
        csum += wolff2d!(s, padd, si, sj)
    end
    ncl = max(1, round(Int, N / (csum / n_eq_cl)))
    Es = zeros(n_meas); Ms = zeros(n_meas)
    for t in 1:n_meas
        if biased
            c = 0
            while c < N
                c += wolff2d!(s, padd, si, sj)
            end
        else
            for _ in 1:ncl
                wolff2d!(s, padd, si, sj)
            end
        end
        Es[t] = energy2d(s); Ms[t] = sum(s)
    end
    return Es, Ms
end

# ------------------------------------------------------------------ 3D
function energy3d(s)
    L = size(s, 1); e = 0
    @inbounds for k in 1:L, j in 1:L, i in 1:L
        e -= s[i, j, k] * (s[mod1(i + 1, L), j, k] + s[i, mod1(j + 1, L), k] + s[i, j, mod1(k + 1, L)])
    end
    return e
end

const NB3 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

function wolff3d!(s, padd, si, sj, sk)
    L = size(s, 1)
    i = rand(1:L); j = rand(1:L); k = rand(1:L); s0 = s[i, j, k]; s[i, j, k] = -s0
    si[1] = i; sj[1] = j; sk[1] = k; top = 1; size_ = 1
    @inbounds while top > 0
        a = si[top]; b = sj[top]; c = sk[top]; top -= 1
        for (di, dj, dk) in NB3
            x = mod1(a + di, L); y = mod1(b + dj, L); z = mod1(c + dk, L)
            if s[x, y, z] == s0 && rand() < padd
                s[x, y, z] = -s0; top += 1; si[top] = x; sj[top] = y; sk[top] = z; size_ += 1
            end
        end
    end
    return size_
end

function run_wolff3d_sweeps(L, beta, n_eq_cl, n_meas, seed)
    Random.seed!(seed)
    N = L^3; s = ones(Int, L, L, L); padd = 1.0 - exp(-2.0 * beta)
    si = zeros(Int, N); sj = zeros(Int, N); sk = zeros(Int, N)
    csum = 0.0
    for _ in 1:n_eq_cl
        csum += wolff3d!(s, padd, si, sj, sk)
    end
    ncl = max(1, round(Int, N / (csum / n_eq_cl)))
    Es = zeros(n_meas); Ms = zeros(n_meas)
    for t in 1:n_meas
        for _ in 1:ncl
            wolff3d!(s, padd, si, sj, sk)
        end
        Es[t] = energy3d(s); Ms[t] = sum(s)
    end
    return Es, Ms
end

# ------------------------------------------------------------------ 통계
"정규화 자기상관 rho(t), t = 0..tmax (반환 배열의 [t+1]번 원소), FFT 사용"
function autocorr(x, tmax)
    x = x .- mean(x); n = length(x)
    f = rfft(vcat(x, zeros(n)))                  # np.fft.rfft(x, 2n)
    acf = irfft(abs2.(f), 2n)[1:tmax+1]
    acf ./= n:-1:(n-tmax)                        # 겹치는 쌍의 수로 나눔
    return acf ./ acf[1]
end

"Sokal 자동 창 (W >= c tau). 규약 tau = 1/2 + sum rho(t). Madras-Sokal 오차."
function tau_int(x; c = 6.0, tmax = min(length(x) ÷ 4, 200000))
    n = length(x); rho = autocorr(x, tmax)
    tau = 0.5; W = tmax
    for w in 1:tmax
        tau += rho[w+1]
        if w >= c * tau
            W = w; break
        end
    end
    return tau, tau * sqrt(2.0 * (2W + 1) / n), W, rho
end

"Flyvbjerg-Petersen 블로킹: 수준별 단순 표준오차 (블록 수 16 이상인 수준까지)"
function blocking(x)
    x = collect(Float64, x); out = Float64[]
    while length(x) >= 16
        n = length(x); push!(out, sqrt(var(x) / n))
        m = 2 * (n ÷ 2); x = 0.5 .* (x[1:2:m] .+ x[2:2:m])
    end
    return out
end

"최소제곱 직선 y = p1 x + p0 (np.polyfit(x, y, 1)과 같은 값). (기울기, 절편)"
function linfit(x, y)
    X = hcat(collect(Float64, x), ones(length(x)))
    p = X \ collect(Float64, y)
    return p[1], p[2]
end
