# equiv_layers.jl — NequIP형 등변 상호작용 블록(Lab 7.19)과 그것을 쌓은 에너지 모형(Lab 7.21)을 Lux로 직접 구현
# 파이썬 76_e3nn_block.py의 e3nn 블록과 같은 구조: 간선 방향의 구면조화(l ≤ 1) x Bessel 기저 위의 반경 MLP
# → 간선마다 가중치가 다른 CG 텐서곱 → 이웃 합 → 자기 상호작용(o3.Linear) + 잔차 → 게이트 비선형.
# 은닉 irreps 16x0e + 8x1o + 8x1e. l = 1 특징은 Cartesian 3-벡터로 둔다(1o: D = Q, 1e: D = det(Q) Q).
# 필요한 CG 경로가 모두 l ≤ 1이라 손으로 쓸 수 있다:
#   0e⊗0e→0e (곱), 0e⊗1o→1o (스칼라 x 벡터), 1o⊗0e→1o, 1o⊗1o→0e (a·b/√3), 1o⊗1o→1e (a×b/√2),
#   1e⊗0e→1e, 1e⊗1o→1o (a×b/√2).
# 경로 수 7, 간선당 가중치 1152, 블록 매개변수 37,760, 모형 59,072개로 e3nn 판과 같다(정규화 상수와 초기화는 다르다).
# 분자 그래프는 밀집 형식(모든 i ≠ j, 간선 순서 (i, j, b))이며, vec_ij = R_i - R_j (이웃 j에서 중심 i로).
using Lux, NNlib, Random, LinearAlgebra
import LuxCore

"Bessel 기저 × 다항식 포락선(p = 6): e_n(r) = √(2/r_c) sin(nπr/r_c)/r · u(r/r_c), u(1) = u'(1) = u''(1) = 0"
function bessel_basis(d, rc, nb; p = 6)
    T = eltype(d); x = reshape(d, 1, :) ./ T(rc)
    env = @. (1 - (p + 1) * (p + 2) / 2 * x^p + p * (p + 2) * x^(p + 1) - p * (p + 1) / 2 * x^(p + 2)) * (x < 1)
    n = T.(1:nb) .* T(pi / rc)
    return sqrt(T(2 / rc)) .* sin.(n .* reshape(d, 1, :)) ./ reshape(d, 1, :) .* env
end

"성분별 외적: a, b는 (3, …) 배열"
function crossb(a, b)
    c = ntuple(_ -> :, ndims(a) - 1)
    a1, a2, a3 = a[1:1, c...], a[2:2, c...], a[3:3, c...]
    b1, b2, b3 = b[1:1, c...], b[2:2, c...], b[3:3, c...]
    return vcat(a2 .* b3 .- a3 .* b2, a3 .* b1 .- a1 .* b3, a1 .* b2 .- a2 .* b1)
end

struct EquivBlock <: LuxCore.AbstractLuxLayer
    nin::NTuple{3,Int}       # 입력 다중도 (0e, 1o, 1e)
    nout::NTuple{3,Int}      # 게이트 뒤 출력 다중도 (0e, 1o, 1e)
    nb::Int                  # Bessel 기저 수
    nh::Int                  # 반경 MLP 은닉 폭
    rc::Float32
    norm::Float32            # 1/√(평균 이웃 수)
    bug::Bool                # true: (1, x/r, y/r, z/r)를 스칼라 4개로 '잘못 표시'한 고전적 실수
end
EquivBlock(nin, nout; nb = 8, nh = 32, rc = 5.0, avg_neigh = 8.0, bug = false) =
    EquivBlock(nin, nout, nb, nh, Float32(rc), Float32(1 / sqrt(avg_neigh)), bug)

"경로 목록 (이름, 입력 다중도, 출력 다중도); 0e 출력 = 스칼라 + 게이트"
function block_paths(b::EquivBlock)
    a0, a1, a2 = b.nin; c0, c1, c2 = b.nout; g = c0 + c1 + c2
    P = b.bug ? [(:ss4, 4a0, g), (:vs4, 4a1, c1), (:ps4, 4a2, c2)] :
        [(:ss, a0, g), (:sv, a0, c1), (:vs, a1, c1), (:vv0, a1, g), (:vv1e, a1, c2), (:ps, a2, c2), (:pv, a2, c1)]
    return filter(p -> p[2] > 0 && p[3] > 0, P)
end
weight_numel(b::EquivBlock) = sum(p[2] * p[3] for p in block_paths(b))

function LuxCore.initialparameters(rng::AbstractRNG, b::EquivBlock)
    a0, a1, a2 = b.nin; c0, c1, c2 = b.nout; g = c0 + c1 + c2
    nrm(o, i) = randn(rng, Float32, o, i) ./ sqrt(Float32(max(i, 1)))     # e3nn: N(0, 1) 가중치를 √fan_in으로 정규화
    return (W1 = nrm(b.nh, b.nb), W2 = nrm(weight_numel(b), b.nh),         # 반경 MLP (편향 없음)
            L0 = nrm(g, a0), L1 = nrm(c1, a1), L2 = nrm(c2, a2))           # 자기 상호작용 (l마다 채널 혼합, 편향 없음)
end
LuxCore.initialstates(::AbstractRNG, ::EquivBlock) = NamedTuple()

"채널 혼합: L (c x a)을 (k, a, …)의 둘째 축에"
lmix(L, x) = size(x, 2) == 0 ? zeros(eltype(x), size(x, 1), size(L, 1), size(x)[3:end]...) :
    permutedims(reshape(L * reshape(permutedims(x, (2, 1, 3:ndims(x)...)), size(x, 2), :), size(L, 1), size(x, 1), size(x)[3:end]...),
                (2, 1, 3:ndims(x)...))

"입력: ((h, geo), ps, st). h = (s: a0 x n x B, v: 3 x a1 x n x B, p: 3 x a2 x n x B), geo = (Y1, Y4, basis, n, B)"
function (b::EquivBlock)(x, ps, st)
    h, geo = x
    n, B = geo.n, geo.B; E = n * n * B; T = eltype(geo.Y1)
    w = ps.W2 * swish.(ps.W1 * geo.basis)                                  # nw x E (대각 간선은 basis = 0이라 0)
    src(y) = reshape(repeat(reshape(y, size(y, 1), size(y, 2), 1, n, B), 1, 1, n, 1, 1), size(y, 1), size(y, 2), E)
    s_e, v_e, p_e = src(reshape(h.s, 1, size(h.s)...)), src(h.v), src(h.p)  # (1, a0, E), (3, a1, E), (3, a2, E): 이웃 j의 특징
    Y1 = reshape(geo.Y1, 3, 1, E)
    a0, a1, a2 = b.nin; c0, c1, c2 = b.nout; g = c0 + c1 + c2
    o0 = zeros(T, 1, g, E); o1 = zeros(T, 3, c1, E); o2 = zeros(T, 3, c2, E)
    off = 0
    for (nm, a, c) in block_paths(b)
        W = reshape(w[off+1:off+a*c, :], a, c, E); off += a * c
        if nm == :ss;       o0 = o0 .+ batched_mul(s_e, W)
        elseif nm == :sv;   o1 = o1 .+ batched_mul(Y1 .* s_e, W)
        elseif nm == :vs;   o1 = o1 .+ batched_mul(v_e, W)
        elseif nm == :vv0;  o0 = o0 .+ batched_mul(sum(v_e .* Y1; dims = 1) ./ sqrt(T(3)), W)
        elseif nm == :vv1e; o2 = o2 .+ batched_mul(crossb(v_e, repeat(Y1, 1, a1, 1)) ./ sqrt(T(2)), W)
        elseif nm == :ps;   o2 = o2 .+ batched_mul(p_e, W)
        elseif nm == :pv;   o1 = o1 .+ batched_mul(crossb(p_e, repeat(Y1, 1, a2, 1)) ./ sqrt(T(2)), W)
        elseif nm == :ss4;  o0 = o0 .+ batched_mul(reshape(reshape(geo.Y4, 4, 1, E) .* s_e, 1, :, E), W)
        elseif nm == :vs4;  o1 = o1 .+ batched_mul(reshape(reshape(geo.Y4, 1, 4, 1, E) .* reshape(v_e, 3, 1, a1, E), 3, :, E), W)
        elseif nm == :ps4;  o2 = o2 .+ batched_mul(reshape(reshape(geo.Y4, 1, 4, 1, E) .* reshape(p_e, 3, 1, a2, E), 3, :, E), W)
        end
    end
    agg(o) = dropdims(sum(reshape(o, size(o, 1), size(o, 2), n, n, B); dims = 4); dims = 4) .* b.norm   # Σ_j → (k, c, n, B)
    pre0 = dropdims(agg(o0); dims = 1) .+ lmix(ps.L0, reshape(h.s, 1, size(h.s)...))[1, :, :, :]
    pre1 = agg(o1) .+ lmix(ps.L1, h.v)
    pre2 = agg(o2) .+ lmix(ps.L2, h.p)
    gates = sigmoid.(pre0[c0+1:end, :, :])
    hn = (s = swish.(pre0[1:c0, :, :]), v = pre1 .* reshape(gates[1:c1, :, :], 1, c1, n, B),
          p = pre2 .* reshape(gates[c1+1:end, :, :], 1, c2, n, B))
    return (hn, geo), st
end

"간선 기하: vec_ij = R_i - R_j, Y1 = √3 r̂ (3 x E), Y4 = (1, r̂) (bug용), Bessel 기저(대각 간선 0)"
function edge_geometry(R::AbstractArray{T,3}; rc = 5.0, nb = 8) where {T}
    n, B = size(R, 2), size(R, 3)
    vec_ = reshape(R, 3, n, 1, B) .- reshape(R, 3, 1, n, B)
    eye = reshape(T.(Matrix(I, n, n)), 1, n, n, 1)
    d = sqrt.(sum(abs2, vec_; dims = 1) .+ eye)                             # 1 x n x n x B
    u = reshape(vec_ ./ d, 3, :)
    mask = repeat(reshape(1 .- eye, 1, n * n), 1, B)                       # 1 x E
    basis = bessel_basis(vec(d), rc, nb) .* mask
    return (Y1 = sqrt(T(3)) .* u, Y4 = vcat(mask, u), basis = basis, n = n, B = B)
end

"종 one-hot → 16 스칼라 → 블록 2개 → 스칼라 판독 → 합. E = scale · Σ_i ε_i + e0"
struct EquivPot{Em,Bk,Ro} <: Lux.AbstractLuxContainerLayer{(:embed, :blocks, :readout)}
    embed::Em
    blocks::Bk
    readout::Ro
    onehot::Matrix{Float32}                                                # 3 x n (H, C, O)
    rc::Float64
    e0::Base.RefValue{Float64}
    scale::Base.RefValue{Float64}
end
function EquivPot(Z; n0 = 16, n1 = 8, n2 = 8, nlayers = 2, rc = 5.0, bug = false)
    sp = Dict(1 => 1, 6 => 2, 8 => 3)
    oh = Float32[sp[z] == k for k in 1:3, z in Z]
    blocks = [EquivBlock(k == 1 ? (n0, 0, 0) : (n0, n1, n2), (n0, n1, n2); rc, bug) for k in 1:nlayers]
    return EquivPot(Dense(3 => n0; use_bias = false), Chain(blocks...), Dense(n0 => 1; use_bias = false), oh, rc, Ref(0.0), Ref(1.0))
end
function node_features(m::EquivPot, R::AbstractArray{T,3}, ps, st) where {T}
    n, B = size(R, 2), size(R, 3)
    s0, _ = m.embed(eltype(ps.embed.weight).(m.onehot), ps.embed, st.embed)   # 16 x n
    h = (s = repeat(reshape(s0, :, n, 1), 1, 1, B), v = zeros(T, 3, 0, n, B), p = zeros(T, 3, 0, n, B))
    geo = edge_geometry(R; rc = m.rc)
    feats = []
    for (k, blk) in enumerate(m.blocks.layers)
        (h, geo), _ = blk((h, geo), ps.blocks[k], st.blocks[k])
        push!(feats, h)
    end
    return feats
end
function (m::EquivPot)(R, ps, st)
    n, B = size(R, 2), size(R, 3)
    s0, _ = m.embed(eltype(ps.embed.weight).(m.onehot), ps.embed, st.embed)
    h = (s = repeat(reshape(s0, :, n, 1), 1, 1, B), v = zeros(eltype(R), 3, 0, n, B), p = zeros(eltype(R), 3, 0, n, B))
    (h, _), _ = m.blocks((h, edge_geometry(R; rc = m.rc)), ps.blocks, st.blocks)
    e, _ = m.readout(reshape(h.s, size(h.s, 1), :), ps.readout, st.readout)
    return vec(sum(reshape(e, n, B); dims = 1)) .* Float32(m.scale[]) .+ Float32(m.e0[]), st
end
