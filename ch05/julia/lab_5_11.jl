# Lab 5.11 (Julia) — 임계점에서 Metropolis와 Wolff의 벽시계 비교
# 파이썬 labs/ising_wolff.py 대응: T = T_c에서 |m|의 tau_int(Sokal 자동 창)와 독립 표본당 벽시계 시간
# t_ind = 2 tau x (sweep 하나의 시간). Metropolis는 체커보드 + 표, Wolff는 스택 기반 "넣을 때 뒤집기".
# 절대 시간은 기계마다 다르고 난수도 numpy와 다르므로, 비교 대상은 지수(z)와 L=128의 비율의 크기다.
# 주의: 이 파일의 tau_int는 파이썬 ising_wolff.py와 똑같이 자기상관을 (n - t)로 나누지 않는다
# (Lab 5.2의 isingkern 판과 다른 점; 창 W가 n보다 훨씬 작아 차이는 무시할 만하다).
using Printf, Random, Statistics, FFTW

const TC = 2.0 / log(1.0 + sqrt(2.0))

"체커보드 Metropolis. 매 sweep 뒤 |m|"
function metro_series!(s, beta, nsweep)
    L = size(s, 1); N = L * L; w4 = exp(-4beta); w8 = exp(-8beta)
    out = zeros(nsweep); M = sum(s)
    @inbounds for n in 1:nsweep
        for c in 0:1, j in 1:L
            jp = j < L ? j + 1 : 1; jm = j > 1 ? j - 1 : L
            for i in (mod(j - 1 + c, 2) + 1):2:L
                ip = i < L ? i + 1 : 1; im = i > 1 ? i - 1 : L
                x = s[i, j] * (s[ip, j] + s[im, j] + s[i, jp] + s[i, jm])
                flip = x <= 0 || rand() < (x == 2 ? w4 : w8)
                if flip
                    M -= 2 * s[i, j]; s[i, j] = -s[i, j]
                end
            end
        end
        out[n] = abs(M) / N
    end
    return out
end

"Wolff 단일 클러스터 갱신. 매 갱신 뒤 |m|과 클러스터 크기. 선형 첨자 k = (i-1) + (j-1) L + 1"
function wolff_series!(s, beta, nstep)
    L = size(s, 1); N = L * L; padd = 1.0 - exp(-2.0beta)   # 결합 활성 확률
    stack = zeros(Int, N); mser = zeros(nstep); cser = zeros(Int, nstep); M = sum(s)
    @inbounds for n in 1:nstep
        k = rand(1:N); s0 = s[k]; s[k] = -s0                # 넣을 때 뒤집기: 각 자리는 한 번만 들어간다
        stack[1] = k; top = 1; size_ = 1
        while top > 0
            k = stack[top]; top -= 1
            i = (k - 1) % L + 1; j = (k - 1) ÷ L + 1
            for (ii, jj) in ((mod1(i + 1, L), j), (mod1(i - 1, L), j), (i, mod1(j + 1, L)), (i, mod1(j - 1, L)))
                if s[ii, jj] == s0 && rand() < padd
                    s[ii, jj] = -s0; top += 1; stack[top] = (jj - 1) * L + ii; size_ += 1
                end
            end
        end
        M -= 2 * s0 * size_
        mser[n] = abs(M) / N; cser[n] = size_
    end
    return mser, cser
end

"Sokal 자동 창: W >= c tau(W)인 가장 작은 W (파이썬 ising_wolff.tau_int와 같은 정규화)"
function tau_int(x; c = 6.0)
    x = x .- mean(x); n = length(x)
    f = rfft(vcat(x, zeros(n))); acf = irfft(abs2.(f), 2n)[1:n]
    rho = acf ./ acf[1]
    tau = 0.5; W = n - 1
    for w in 1:n-1
        tau += rho[w+1]
        if w >= c * tau
            W = w; break
        end
    end
    return tau, tau * sqrt(2 * (2W + 1) / n)
end

linslope(x, y) = (hcat(x, ones(length(x))) \ y)[1]

function main()
    beta = 1 / TC
    Random.seed!(11); s = ones(Int, 8, 8); metro_series!(s, beta, 10); wolff_series!(s, beta, 10)   # JIT 예열
    println("=== T = T_c = 2.2692, observable |m|; tau in sweeps; t_ind = 2 tau x (wall time per sweep) ===")
    println("   L | Metropolis: tau_int     t_ind[ms] |  Wolff: <C>/N   tau_int     t_ind[ms]")
    Ls = Int[]; tauMs = Float64[]; tauWs = Float64[]; tindM = 0.0; tindW = 0.0
    for (L, nsw) in ((8, 200000), (16, 200000), (32, 200000), (64, 200000), (128, 120000))
        N = L * L
        Random.seed!(100 + L); s = ones(Int, L, L)
        metro_series!(s, beta, 5000)                                   # 평형화
        t0 = time(); mm = metro_series!(s, beta, nsw); tm = (time() - t0) / nsw
        tauM, eM = tau_int(mm)
        Random.seed!(200 + L); s = ones(Int, L, L)
        wolff_series!(s, beta, 2000)
        nst = 200000
        t0 = time(); mw, cw = wolff_series!(s, beta, nst); tw_step = (time() - t0) / nst
        cfrac = mean(cw) / N                                           # 클러스터 갱신당 sweep 환산
        tauW_steps, eW = tau_int(mw)
        tauW = tauW_steps * cfrac; eWs = eW * cfrac; tw_sweep = tw_step / cfrac
        tindM = 2tauM * tm; tindW = 2tauW * tw_sweep
        push!(Ls, L); push!(tauMs, tauM); push!(tauWs, tauW)
        @printf(" %3d |        %8.1f+/-%6.1f  %9.3f |  %6.3f   %6.2f+/-%4.2f  %9.4f\n",
                L, tauM, eM, tindM * 1e3, cfrac, tauW, eWs, tindW * 1e3)
    end
    zM = linslope(log.(Ls[2:end]), log.(tauMs[2:end])); zW = linslope(log.(Ls[2:end]), log.(tauWs[2:end]))
    @printf("fit L=16..128: tau_Metropolis ~ L^%.2f,  tau_Wolff(sweeps) ~ L^%.2f\n", zM, zW)
    @printf("L=128: Wolff delivers an independent sample %.0fx faster in wall time\n", tindM / tindW)
end

main()
