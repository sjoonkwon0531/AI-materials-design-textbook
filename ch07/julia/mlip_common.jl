# mlip_common.jl — 에탄올 MLIP Lab(7.8–7.12, 7.16, 7.21, 7.22)의 공용 부분 (Lux.jl + Zygote, 중첩 AD로 힘 학습)
# 배치 약속(column-major): 좌표 R은 3 x n x B(열 = 원자), 힘도 같은 모양. 파이썬 (B, n, 3)의 전치다.
# 모든 분자 그래프는 작은 분자용 '밀집' 형식(i, j 쌍 전체 + 대각 마스크)으로 쓴다. 파이썬 72b_schnet.py의
# dense_forward와 같은 함수이고, scatter 없이 reshape·broadcast·행렬곱만 쓰므로 Zygote와 중첩 AD가 편하다.
using Lux, Zygote, ForwardDiff, NNlib, NPZ, Random, Statistics, LinearAlgebra, Printf
import Optimisers, Functors

# ---------------------------------------------------------------- 데이터
"ethanol GFN2-xTB 궤적(T = 300, 600, 1000 K): R, F는 3 x 9 x 프레임, E는 eV"
function load_ethanol(T::Int = 300)
    d = npzread(joinpath(DATA, "ethanol_xtb_$(T)K.npz"))
    return permutedims(d["R"], (3, 2, 1)), Vector{Int}(d["Z"]), Vector{Float64}(d["E"]), permutedims(d["F"], (3, 2, 1))
end

"누수를 의식한 분할(6.7절): 풀 0–1999에서 검증 100, 나머지 학습; 시험은 2 ps 뒤의 2200–2999.
파이썬과 같은 순서를 쓰려면 data/splits/ethanol_pool_perm.csv(np.random.default_rng(0).permutation(2000))를 읽는다"
function split_300K(; from_python = true)
    pool = from_python ? read_idx("ethanol_pool_perm.csv") : randperm(Xoshiro(0), 2000)
    return pool[101:end], pool[1:100], collect(2201:3000)
end

# ---------------------------------------------------------------- 기하와 기저
"vec[:, i, j, b] = R_j - R_i, 거리 r[i, j, b](대각은 sqrt(0)을 피하려 1을 더함), 비대각 마스크 offd (n x n x 1)"
function pair_geometry(R::AbstractArray{T,3}) where {T}
    n, B = size(R, 2), size(R, 3)
    vec = reshape(R, 3, 1, n, B) .- reshape(R, 3, n, 1, B)
    eye = reshape(T.(Matrix(I, n, n)), n, n, 1)
    r = sqrt.(dropdims(sum(abs2, vec; dims = 1); dims = 1) .+ eye)
    return vec, r, 1 .- eye
end
cosine_cutoff(r, rc) = 0.5 .* (cos.(pi .* r ./ rc) .+ 1) .* (r .< rc)
"가우스 RBF: (n_rbf, 거리 개수)"
gaussian_rbf(r, mu, gamma) = exp.(-gamma .* (reshape(r, 1, :) .- mu) .^ 2)
"shifted softplus ln(0.5 eˣ + 0.5) (SchNet 활성화, ssp(0) = 0)"
ssp(x) = softplus(x) - oftype(x, log(2))

"PyTorch nn.Linear와 같은 분포의 Lux Dense (가중치 (out, in), 편향 U(±1/√fan_in))"
tu(nin) = (rng, dims...) -> (rand(rng, Float32, dims...) .* 2 .- 1) ./ sqrt(Float32(nin))
ldense(nin, nout, act = identity; bias = true) = Dense(nin => nout, act; init_weight = tu(nin), init_bias = tu(nin), use_bias = bias)

# ---------------------------------------------------------------- SchNet (밀집 형식)
"연속 필터 합성곱 상호작용 블록. 입력·출력이 (h, rbf, fc) 튜플이라 Lux.Chain으로 쌓인다. h: F x n x B"
struct CFConv{A,Bl,C} <: Lux.AbstractLuxContainerLayer{(:filt, :in2f, :out)}
    filt::A
    in2f::Bl
    out::C
end
CFConv(nf::Int, nrbf::Int) = CFConv(Chain(ldense(nrbf, nf, ssp), ldense(nf, nf)), ldense(nf, nf; bias = false),
                                    Chain(ldense(nf, nf, ssp), ldense(nf, nf)))
function (c::CFConv)(x, ps, st)
    h, rbf, fc = x
    F, n, B = size(h)
    W, _ = c.filt(rbf, ps.filt, st.filt)                                # F x (n n B)
    W = reshape(W, F, n, n, B) .* reshape(fc, 1, n, n, B)                # W[f, i, j, b] f_c(r_ij)
    a, _ = c.in2f(reshape(h, F, :), ps.in2f, st.in2f)
    m = dropdims(sum(W .* reshape(a, F, 1, n, B); dims = 3); dims = 3)   # m_i = Σ_j W_ij ⊙ (W₁ h_j)
    o, _ = c.out(reshape(m, F, :), ps.out, st.out)
    return (h .+ reshape(o, F, n, B), rbf, fc), st                        # 잔차 연결
end

"""밀집 SchNet: E = scale · Σ_i ε_i (pernorm이면 /n) + e0. tokens는 Embedding 행 번호(1-based).
파이썬 72b(원자번호를 0-based 행으로)와 맞추려면 tokens = Z .+ 1, maxz = 100."""
struct DenseSchNet{E,Bk,H} <: Lux.AbstractLuxContainerLayer{(:emb, :blocks, :head)}
    emb::E
    blocks::Bk
    head::H
    tokens::Vector{Int}
    mu::Vector{Float32}
    gamma::Float32
    rc::Float32
    pernorm::Bool
    e0::Base.RefValue{Float64}
    scale::Base.RefValue{Float64}
end
function DenseSchNet(tokens; nf = 32, nrbf = 20, rc = 5.0, ninter = 3, maxz = 100, gamma = nothing, pernorm = false)
    mu = collect(range(0.0f0, Float32(rc); length = nrbf))
    g = gamma === nothing ? Float32(0.5 / Float64(mu[2] - mu[1])^2) : Float32(gamma)   # torch와 같은 반올림 순서
    return DenseSchNet(Embedding(maxz => nf), Chain([CFConv(nf, nrbf) for _ in 1:ninter]...),
                       Chain(ldense(nf, nf ÷ 2, ssp), ldense(nf ÷ 2, 1)), tokens, mu, g, Float32(rc), pernorm, Ref(0.0), Ref(1.0))
end
function atom_features(m::DenseSchNet, R::AbstractArray{T,3}, ps, st) where {T}
    n, B = size(R, 2), size(R, 3)
    _, r, offd = pair_geometry(R)
    fc = cosine_cutoff(r, m.rc) .* offd
    rbf = gaussian_rbf(r, m.mu, m.gamma)                                # Float32 상수 (Dual과도 승격된다)
    h0, _ = m.emb(m.tokens, ps.emb, st.emb)                              # F x n
    h = repeat(h0, 1, 1, B)
    (h, _, _), _ = m.blocks((h, rbf, fc), ps.blocks, st.blocks)
    return h
end
function (m::DenseSchNet)(R, ps, st)
    h = atom_features(m, R, ps, st)
    F, n, B = size(h)
    eps, _ = m.head(reshape(h, F, :), ps.head, st.head)
    E = vec(sum(reshape(eps, n, B); dims = 1)) .* Float32(m.scale[] / (m.pernorm ? n : 1)) .+ Float32(m.e0[])
    return E, st
end

# ---------------------------------------------------------------- ACSF + 원소별 MLP (BPNN, 6장 식 (6.3)–(6.4))
struct ACSFSpec
    rc::Float64; eta2::Float64; rs::Vector{Float64}; eta4::Float64
    zetas::Vector{Float64}; lams::Vector{Float64}
    species::Vector{Int}
    onehot::Matrix{Float64}          # n x S
    pairmask::Array{Float64,3}       # n x n x P (무순서 종 쌍)
end
function ACSFSpec(Z; rc = 5.0, eta2 = 4.0, rs = collect(range(0.8, 4.4; length = 8)), eta4 = 0.005, zetas = [1.0, 4.0], lams = [1.0, -1.0])
    sp = sort(unique(Z)); n = length(Z)
    pairs = [(a, b) for (ia, a) in enumerate(sp) for b in sp[ia:end]]
    M = zeros(n, n, length(pairs))
    for j in 1:n, k in 1:n, (p, (a, b)) in enumerate(pairs)
        (Z[j], Z[k]) in ((a, b), (b, a)) && (M[j, k, p] = 1.0)
    end
    return ACSFSpec(rc, eta2, rs, eta4, zetas, lams, sp, [Float64(Z[j] == s) for j in 1:n, s in sp], M)
end
n_acsf(a::ACSFSpec) = length(a.species) * length(a.rs) + size(a.pairmask, 3) * length(a.zetas) * length(a.lams)

"G (n_out x n x B): 종마다 G2 블록(동경 기저 안쪽 순서), 이어서 종 쌍마다 (ζ, λ) G4 블록. 파이썬 ACSF와 같은 식"
function acsf(R::AbstractArray{T,3}, a::ACSFSpec) where {T}
    n, B = size(R, 2), size(R, 3); S = length(a.species); P = size(a.pairmask, 3); nrs = length(a.rs)
    rc, eta2, eta4, rs = Float32(a.rc), Float32(a.eta2), Float32(a.eta4), Float32.(a.rs)   # Float32 상수로
    vec, r, offd = pair_geometry(R)
    fc = cosine_cutoff(r, rc) .* offd                                                     # i j b
    g2 = exp.(-eta2 .* (reshape(r, 1, n, n, B) .- rs) .^ 2) .* reshape(fc, 1, n, n, B)    # rs i j b
    G2 = reshape(reshape(permutedims(g2, (1, 2, 4, 3)), :, n) * Float32.(a.onehot), nrs, n, B, S)   # rs i b s
    G2 = reshape(permutedims(G2, (1, 4, 2, 3)), nrs * S, n, B)
    u = vec ./ reshape(r, 1, n, n, B)
    cosv = dropdims(sum(reshape(u, 3, n, n, 1, B) .* reshape(u, 3, n, 1, n, B); dims = 1); dims = 1)   # i j k b
    r2 = r .^ 2
    rad = exp.(-eta4 .* (reshape(r2, n, n, 1, B) .+ reshape(r2, n, 1, n, B) .+ reshape(r2, 1, n, n, B)))
    cut = reshape(fc, n, n, 1, B) .* reshape(fc, n, 1, n, B) .* reshape(fc, 1, n, n, B)
    base = rad .* cut .* reshape(offd, 1, n, n, 1)
    Mjk = reshape(Float32.(a.pairmask), n * n, P)
    G4 = map([(z, l) for z in a.zetas for l in a.lams]) do (z, l)
        ang = Float32(2^(1 - z)) .* max.(1 .+ Float32(l) .* cosv, 0) .^ Float32(z) .* base                         # i j k b
        reshape(permutedims(reshape(reshape(permutedims(ang, (1, 4, 2, 3)), n * B, n * n) * Mjk, n, B, P), (3, 1, 2)), 1, P, n, B)
    end
    G4 = reshape(cat(G4...; dims = 1), length(a.zetas) * length(a.lams) * P, n, B)
    return cat(G2, G4; dims = 1)
end

"원소별 MLP(48-48-1, SiLU) 세 개(에탄올의 H, C, O). E = Σ_i ε_{Z_i}(G_i) + e0"
struct BPNN{A,Bn,C} <: Lux.AbstractLuxContainerLayer{(:net1, :net2, :net3)}
    net1::A
    net2::Bn
    net3::C
    spec::ACSFSpec
    Z::Vector{Int}
    mu::Base.RefValue{Vector{Float32}}
    sd::Base.RefValue{Vector{Float32}}
    e0::Base.RefValue{Float64}
end
function BPNN(Z; hidden = 48)
    spec = ACSFSpec(Z); d = n_acsf(spec)
    @assert length(spec.species) == 3 "이 짧은 판은 종 세 개(H, C, O)만 다룬다"
    net() = Chain(ldense(d, hidden, swish), ldense(hidden, hidden, swish), ldense(hidden, 1))
    return BPNN(net(), net(), net(), spec, Z, Ref(zeros(Float32, d)), Ref(ones(Float32, d)), Ref(0.0))
end
"학습 구조로 G의 평균과 표준편차(불편, + 1e-6)를 맞춘다"
function fit_scaler!(m::BPNN, R)
    G = reshape(acsf(Float32.(R), m.spec), n_acsf(m.spec), :)
    m.mu[] = vec(mean(G; dims = 2)); m.sd[] = vec(std(G; dims = 2)) .+ 1.0f-6
    return m
end
function (m::BPNN)(R::AbstractArray{T,3}, ps, st) where {T}
    G = (acsf(R, m.spec) .- m.mu[]) ./ m.sd[]
    d, n, B = size(G)
    E = fill(Float32(m.e0[]), B)
    for (k, (net, p, s)) in enumerate(((m.net1, ps.net1, st.net1), (m.net2, ps.net2, st.net2), (m.net3, ps.net3, st.net3)))
        idx = findall(==(m.spec.species[k]), m.Z)
        e, _ = net(reshape(G[:, idx, :], d, :), p, s)
        E = E .+ vec(sum(reshape(e, length(idx), B); dims = 1))
    end
    return E, st
end

# ---------------------------------------------------------------- 에너지·힘, 학습, 평가
"E (B)와 F = -∂E/∂R (3 x n x B). 바깥 Zygote.gradient 안에서 불리면 Lux가 안쪽 기울기를 중첩 AD로 처리한다"
function energy_forces(model, ps, st, R)
    sm = StatefulLuxLayer{true}(model, ps, st)
    return sm(R), -only(Zygote.gradient(sum ∘ sm, R))
end

"균일 무작위 회전 n개 (3 x 3 x n): 가우스 행렬의 QR, 부호 보정, det = +1"
function random_rotations(rng, n)
    Q = zeros(3, 3, n)
    for k in 1:n
        F = qr(randn(rng, 3, 3)); M = Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
        det(M) < 0 && (M[:, 1] .*= -1)
        Q[:, :, k] = M
    end
    return Q
end
"배치 회전: X (3 x n x B)의 각 분자에 Q[:, :, b]를 곱한다"
rotate_batch(Q, X) = batched_mul(eltype(X).(Q), X)

"Kabsch: P, Q (3 x k, 열 = 점)에 대해 M (P - c_P) ≈ Q - c_Q인 고유 회전 M (det M = +1)"
function kabsch(P, Q)
    P = P .- mean(P; dims = 2); Q = Q .- mean(Q; dims = 2)
    F = svd(P * Q')
    d = sign(det(F.V * F.U'))
    return F.V * Diagonal([1.0, 1.0, d]) * F.U'
end

"Adam + cosine, 손실 lamE <ΔE²> + <|ΔF|²>/3 (성분 평균). 검증 E_MAE + F_MAE가 가장 좋은 매개변수를 돌려준다"
function train_ef(model, ps, st, R, E, F, Rv, Ev, Fv; steps = 3000, bs = 16, lr = 3e-3, lamE = 0.1, seed = 0,
                  augment = false, eta_min = 0.0, evalevery = 100, history = nothing, score = (e, f) -> e + f)
    rng = Xoshiro(seed)
    R, F, E = Float32.(R), Float32.(F), Float32.(E)
    hasproperty(model, :e0) && (model.e0[] = mean(E))
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    best = (Inf, deepcopy(ps)); n = length(E)
    for s in 1:steps
        idx = randperm(rng, n)[1:min(bs, n)]
        Rb, Eb, Fb = R[:, :, idx], E[idx], F[:, :, idx]
        if augment                                                        # 입력과 힘 레이블을 함께 회전
            Q = random_rotations(rng, length(idx))
            Rb = rotate_batch(Q, Rb .- mean(Rb; dims = 2)); Fb = rotate_batch(Q, Fb)
        end
        Optimisers.adjust!(opt, cosine_lr(lr, s - 1, steps; eta_min))
        g = only(Zygote.gradient(ps) do p
            Ep, Fp = energy_forces(model, p, st, Rb)
            lamE * mean(abs2, Ep .- Eb) + mean(abs2, Fp .- Fb)
        end)
        opt, ps = Optimisers.update!(opt, ps, g)
        if s % evalevery == 0 || s == steps
            eE, eF = evaluate(model, ps, st, Rv, Ev, Fv)
            history === nothing || push!(history, (s, eE * 1e3, eF * 1e3))
            score(eE, eF) < best[1] && (best = (score(eE, eF), deepcopy(ps)))
        end
    end
    return best[2]
end

"(E MAE, F MAE) — 배치로 나눠 평가 (eV, eV/Å)"
function evaluate(model, ps, st, R, E, F; bs = 200)
    eE, eF = Float64[], Float64[]
    for k in 1:bs:size(R, 3)
        j = k:min(k + bs - 1, size(R, 3))
        Ep, Fp = energy_forces(model, ps, st, Float32.(R[:, :, j]))
        append!(eE, abs.(Float64.(Ep) .- E[j])); append!(eF, vec(abs.(Fp .- F[:, :, j])))
    end
    return mean(eE), mean(eF)
end

count_params(ps) = Lux.parameterlength(ps)
"대칭 점검용 Float64 매개변수 (기계 정밀도 비교는 Float64에서 한다)"
to64(ps) = Functors.fmap(x -> x isa AbstractArray{<:AbstractFloat} ? Float64.(x) : x, ps)

# ---------------------------------------------------------------- 파이썬에서 학습한 SchNet 가중치 옮기기 (PythonCall)
"""72b_schnet_ethanol_N*.pt(SchNetMol state_dict)를 DenseSchNet(tokens = Z .+ 1, maxz = 100)의 ps로 옮긴다.
nn.Linear.weight는 (out, in)으로 Lux와 같고, nn.Embedding.weight (100, F)는 Lux의 (F, 100)로 전치한다.
torch Sequential의 0, 2번 층(1번은 SSP)이 Lux Chain의 layer_1, layer_2다."""
function load_torch_schnet!(ps, path; model = nothing)
    torch = pyimport("torch")
    sd = torch.load(path; map_location = "cpu")
    g(k) = pyconvert(Array{Float32}, sd[k].numpy())
    model === nothing || (model.mu .= g("net.rbf.mu"))                     # torch.linspace와 마지막 자리까지 같게
    ps.emb.weight .= permutedims(g("net.emb.weight"))
    for (b, blk) in enumerate(values(ps.blocks))
        pre = "net.blocks.$(b - 1)."
        blk.filt.layer_1.weight .= g(pre * "filt.0.weight"); blk.filt.layer_1.bias .= g(pre * "filt.0.bias")
        blk.filt.layer_2.weight .= g(pre * "filt.2.weight"); blk.filt.layer_2.bias .= g(pre * "filt.2.bias")
        blk.in2f.weight .= g(pre * "in2f.weight")
        blk.out.layer_1.weight .= g(pre * "out.0.weight"); blk.out.layer_1.bias .= g(pre * "out.0.bias")
        blk.out.layer_2.weight .= g(pre * "out.2.weight"); blk.out.layer_2.bias .= g(pre * "out.2.bias")
    end
    ps.head.layer_1.weight .= g("net.head.0.weight"); ps.head.layer_1.bias .= g("net.head.0.bias")
    ps.head.layer_2.weight .= g("net.head.2.weight"); ps.head.layer_2.bias .= g("net.head.2.bias")
    return ps
end
