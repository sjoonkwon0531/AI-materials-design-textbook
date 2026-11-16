# bcp_common.jl — §5.8 블록공중합체 DPD 사례 연구의 공용 도우미. 파이썬 labs/bcp_common.py 대응.
# 비드 번호 i = (c-1) N + k (사슬 c, 사슬 안 위치 k, 모두 1-based). 좌표는 3 x n.
using Random, Statistics, LinearAlgebra

"무작위 보행 사슬 nch개 A_{fN} B_{(1-f)N}. (x, v, typ, bonds[2 x nb])"
function build_melt(nch, N, fA, box; b0 = 0.8)
    n = nch * N; NA = round(Int, fA * N)
    x = zeros(3, n); typ = zeros(Int, n); bonds = zeros(Int, 2, nch * (N - 1)); nb = 0
    for c in 1:nch
        p = rand(3) .* box
        for k in 1:N
            i = (c - 1) * N + k
            if k > 1
                d = randn(3); p = p .+ b0 .* d ./ norm(d)
                nb += 1; bonds[1, nb] = i - 1; bonds[2, nb] = i
            end
            x[:, i] = mod.(p, box); typ[i] = k <= NA ? 0 : 1
        end
    end
    v = randn(3, n); v .-= mean(v; dims = 2)
    return x, v, typ, bonds
end

"반공간 정수 q벡터 2 pi n / L, 0 < |n| <= nmax (정육면체 상자). K x 3 행렬"
function qvectors(box, nmax)
    L = box[1]; ns = Vector{Float64}[]
    for a in -nmax:nmax, b in -nmax:nmax, c in -nmax:nmax     # 파이썬과 같은 순서 (a 바깥)
        ((a, b, c) > (0, 0, 0) && a^2 + b^2 + c^2 <= nmax^2) && push!(ns, [a, b, c])
    end
    return 2pi .* permutedims(hcat(ns...)) ./ L
end

"S_psi(q) = |sum_j e_j exp(i q.r_j)|^2 / n, e_j = +1 (A), -1 (B), 각 q 행마다"
function sq_psi(x, e, qv)
    n = size(x, 2); out = zeros(size(qv, 1))
    @inbounds for k in 1:size(qv, 1)
        re = 0.0; im = 0.0
        for j in 1:n
            ph = qv[k, 1] * x[1, j] + qv[k, 2] * x[2, j] + qv[k, 3] * x[3, j]
            re += e[j] * cos(ph); im += e[j] * sin(ph)
        end
        out[k] = (re * re + im * im) / n
    end
    return out
end

"|q|가 같은 껍질끼리 평균 (|q|를 소수 여섯째 자리로 반올림해 묶음)"
function radial(qv, s)
    qm = round.(sqrt.(vec(sum(qv .^ 2; dims = 2))); digits = 6); uq = sort(unique(qm))
    return uq, [mean(s[qm .== u]) for u in uq]
end

"사슬을 결합을 따라 펼친다. 3 x N x nch"
function unwrap_chains(x, nch, N, box)
    y = copy(reshape(x, 3, N, nch))
    for c in 1:nch, k in 2:N
        d = y[:, k, c] .- y[:, k-1, c]; d .-= box .* round.(d ./ box); y[:, k, c] = y[:, k-1, c] .+ d
    end
    return y
end

function chain_stats(x, nch, N, box)
    y = unwrap_chains(x, nch, N, box)
    Re2 = mean(sum((y[:, N, :] .- y[:, 1, :]) .^ 2; dims = 1))
    cm = mean(y; dims = 2); Rg2 = mean(sum((y .- cm) .^ 2; dims = 1))
    b2 = mean(sum((y[:, 2:end, :] .- y[:, 1:end-1, :]) .^ 2; dims = 1))
    return [Re2, Rg2, b2]
end

"단일 사슬 블록 형태인자 s_AA, s_BB, s_AB (비드당, 즉 N으로 나눔), |q|마다 무작위 방향 nrand개로 방향 평균"
function intrachain_sq(x, nch, N, NA, box, qs; nrand = 24)
    y = unwrap_chains(x, nch, N, box)
    dirs = randn(3, nrand); dirs ./= sqrt.(sum(dirs .^ 2; dims = 1))
    proj = [sum(dirs[:, k] .* y[:, nn, c]) for k in 1:nrand, c in 1:nch, nn in 1:N]   # (nrand, nch, N)
    out = zeros(length(qs), 3)
    for (iq, q) in enumerate(qs)
        E = cis.(q .* proj)
        ea = dropdims(sum(E[:, :, 1:NA]; dims = 3); dims = 3); eb = dropdims(sum(E[:, :, NA+1:end]; dims = 3); dims = 3)
        out[iq, 1] = mean(abs2.(ea)) / N; out[iq, 2] = mean(abs2.(eb)) / N
        out[iq, 3] = mean(real.(ea .* conj.(eb))) / N
    end
    return out
end

"비압축 RPA (Leibler), psi = phi_A - phi_B 비드당: S_psi = 4 / (Gamma - 2 chi)"
function rpa_spsi(saa, sbb, sab, chi)
    gam = (saa .+ sbb .+ 2 .* sab) ./ (saa .* sbb .- sab .^ 2)
    return 4.0 ./ (gam .- 2chi)
end

"선형 보간 (np.interp)"
function interp1(xq, xs, ys)
    map(xq) do x
        k = clamp(searchsortedlast(xs, x), 1, length(xs) - 1)
        ys[k] + (ys[k+1] - ys[k]) * (x - xs[k]) / (xs[k+1] - xs[k])
    end
end

"구간 [a, b]에서 1변수 최소화 (황금분할; scipy minimize_scalar(method='bounded')와 같은 역할)"
function fminbound(f, a, b; tol = 1e-6)
    g = (sqrt(5) - 1) / 2; c = b - g * (b - a); d = a + g * (b - a); fc = f(c); fd = f(d)
    while b - a > tol
        if fc < fd
            b, d, fd = d, c, fc; c = b - g * (b - a); fc = f(c)
        else
            a, c, fc = c, d, fd; d = a + g * (b - a); fd = f(d)
        end
    end
    return (a + b) / 2
end
