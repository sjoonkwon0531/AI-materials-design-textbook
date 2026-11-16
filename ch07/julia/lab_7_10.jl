# lab_7_10.jl — Lab 7.10: 좌표를 그대로 받는 Transformer 대 불변 GNN (labs/73a_attn.py)
# 토큰 = Emb(Z) + Linear(x - 무게중심), pre-LN 자기주의 블록 2개(d 64, 머리 4), E = Σ_i MLP(토큰_i), F = -∂E/∂R.
# 다중 머리 주의는 NNlib.batched_mul과 softmax로 직접 썼다(머리·배치를 셋째 축에 접는다).
# 학습 다섯 번은 파이썬 판처럼 각 수 분씩 걸린다. 결과는 같은 경향(통계적 일치)으로 비교한다.
include("ch7_common.jl")
include("mlip_common.jl")

"pre-LN 자기주의 블록: x + O(Attn(LN x)), 그다음 x + FF(LN x). x: d x n x B"
struct AttnBlock{L1,L2,Q,O,F} <: Lux.AbstractLuxContainerLayer{(:ln1, :ln2, :qkv, :o, :ff)}
    ln1::L1
    ln2::L2
    qkv::Q
    o::O
    ff::F
    heads::Int
end
AttnBlock(d = 64, heads = 4, ff = 128) = AttnBlock(LayerNorm((d,); dims = 1), LayerNorm((d,); dims = 1), ldense(d, 3d), ldense(d, d),
                                                   Chain(ldense(d, ff, swish), ldense(ff, d)), heads)
function (b::AttnBlock)(x, ps, st)
    d, n, B = size(x); h = b.heads; dk = d ÷ h
    y, _ = b.ln1(x, ps.ln1, st.ln1)
    qkv, _ = b.qkv(reshape(y, d, :), ps.qkv, st.qkv)
    qkv = reshape(qkv, dk, h, 3, n, B)                                   # torch reshape(B, n, 3, h, dk)의 전치
    part(k) = reshape(permutedims(qkv[:, :, k, :, :], (1, 3, 2, 4)), dk, n, h * B)   # dk x n x (h B)
    q, kk, v = part(1), part(2), part(3)
    A = softmax(batched_mul(batched_transpose(kk), q) ./ sqrt(Float32(dk)); dims = 1)   # A[j, i] = softmax_j(q_i·k_j/√dk)
    att = batched_mul(v, A)                                              # dk x n x (h B)
    att = reshape(permutedims(reshape(att, dk, n, h, B), (1, 3, 2, 4)), d, n * B)
    o, _ = b.o(att, ps.o, st.o)
    x = x .+ reshape(o, d, n, B)
    y2, _ = b.ln2(x, ps.ln2, st.ln2)
    f, _ = b.ff(reshape(y2, d, :), ps.ff, st.ff)
    return x .+ reshape(f, d, n, B), st
end

struct XYZTransformer{E,P,Bk,H} <: Lux.AbstractLuxContainerLayer{(:emb, :pos, :blocks, :head)}
    emb::E
    pos::P
    blocks::Bk
    head::H
    tokens::Vector{Int}
    e0::Base.RefValue{Float64}
end
XYZTransformer(Z; d = 64, nl = 2) = XYZTransformer(Embedding(20 => d), ldense(3, d), Chain([AttnBlock(d) for _ in 1:nl]...),
                                                   Chain(LayerNorm((d,); dims = 1), ldense(d, d, swish), ldense(d, 1)), Z .+ 1, Ref(0.0))
function (m::XYZTransformer)(R, ps, st)
    n, B = size(R, 2), size(R, 3)
    e, _ = m.emb(m.tokens, ps.emb, st.emb)
    p, _ = m.pos(reshape(R .- mean(R; dims = 2), 3, :), ps.pos, st.pos)  # 무게중심 기준 좌표 → 이동 불변, 회전 대칭은 없음
    x, _ = m.blocks(reshape(p, :, n, B) .+ e, ps.blocks, st.blocks)
    y, _ = m.head(x, ps.head, st.head)
    return vec(sum(reshape(y, n, B); dims = 1)) .+ Float32(m.e0[]), st
end

"27개 중심화 좌표 → MLP (이동 불변만)"
struct XYZMLP{C} <: Lux.AbstractLuxContainerLayer{(:f,)}
    f::C
    e0::Base.RefValue{Float64}
end
XYZMLP() = XYZMLP(Chain(ldense(27, 128, swish), ldense(128, 128, swish), ldense(128, 1)), Ref(0.0))
(m::XYZMLP)(R, ps, st) = (vec(first(m.f(reshape(R .- mean(R; dims = 2), 27, :), ps.f, st.f))) .+ Float32(m.e0[]), st)

function main()
    R, Z, E, F = load_ethanol(300)
    tr, va, te = split_300K()
    ref = R[:, 1:3, 1]
    Rc, Fc = similar(R), similar(F); ang = zeros(size(R, 3))
    for k in axes(R, 3)                                                  # 정준 좌표계: C, C, O를 프레임 0에 Kabsch로 맞춤
        M = kabsch(R[:, 1:3, k], ref)
        Rc[:, :, k] = M * (R[:, :, k] .- mean(R[:, :, k]; dims = 2)); Fc[:, :, k] = M * F[:, :, k]
        ang[k] = rad2deg(acos(clamp((tr_(M) - 1) / 2, -1, 1)))
    end
    @printf("orientation of raw frames vs frame 0: median %.0f deg\n", median(ang))
    Qte = random_rotations(Xoshiro(7), length(te))
    idx = tr[1:1000]
    runs = [("Transformer, raw frames", () -> XYZTransformer(Z), :raw, false), ("MLP(xyz), canonical", XYZMLP, :canon, false),
            ("Transformer, canonical", () -> XYZTransformer(Z), :canon, false), ("Transformer, canonical +aug", () -> XYZTransformer(Z), :canon, true),
            ("SchNet, raw frames", () -> DenseSchNet(Z .+ 1; nf = 32, nrbf = 20, rc = 5.0, ninter = 3), :raw, false)]
    @printf("%-28s %7s %7s | %15s %7s | %12s %13s\n", "model", "E MAE", "F MAE", "rotated test: E", "F", "|E(Qx)-E(x)|", "|F(Qx)-QF(x)|")
    for (name, mk, frame, aug) in runs
        Rs, Fs = frame == :raw ? (R, F) : (Rc, Fc)
        model = mk(); ps, st = Lux.setup(Xoshiro(0), model)
        h = Tuple{Int,Float64,Float64}[]
        ps = train_ef(model, ps, st, Rs[:, :, idx], E[idx], Fs[:, :, idx], Rs[:, :, va], E[va], Fs[:, :, va];
                      steps = 3000, bs = 16, lr = 3e-3, augment = aug, history = h)
        Rt, Ft = Rs[:, :, te], Fs[:, :, te]
        eE, eF = evaluate(model, ps, st, Rt, E[te], Ft)
        Rr, Fr = rotate_batch(Qte, Rt), rotate_batch(Qte, Ft)
        eEr, eFr = evaluate(model, ps, st, Rr, E[te], Fr)
        E0, F0 = energy_forces(model, ps, st, Float32.(Rt)); E1, F1 = energy_forces(model, ps, st, Float32.(Rr))
        @printf("%-28s %7.1f %7.1f | %15.1f %7.1f | %12.1f %13.1f   (%d params)\n", name, eE * 1e3, eF * 1e3, eEr * 1e3, eFr * 1e3,
                mean(abs.(E1 .- E0)) * 1e3, mean(abs.(F1 .- rotate_batch(Qte, F0))) * 1e3, count_params(ps))
    end
end
tr_(M) = M[1, 1] + M[2, 2] + M[3, 3]

main()
