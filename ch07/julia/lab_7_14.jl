# lab_7_14.jl — Lab 7.14: 등변 합성곱 층을 손으로 만들고 검증하기 (labs/73b_tp_layer.py의 e3nn 층을 직접 구현)
# e3nn에 견줄 만한 Julia 포트를 찾지 못해, 이 Lab의 층(입력 8x0e + 8x1o, 구면조화 l ≤ 2, 출력 8x0e + 8x1o + 8x2e,
# 게이트)을 so3.jl의 손으로 쓴 CG 경로 일곱 개로 구현했다. 경로별 정규화 상수는 e3nn과 다르지만(등변성과는 무관),
# 경로 수·간선당 가중치 수·매개변수 수는 e3nn과 같게 맞췄다. e3nn 자체를 부르는 PythonCall 패턴은 파일 끝에 있다.
include("so3.jl")
using Printf, Statistics, NNlib, NPZ

# ---------------------------------------------------------------- Part A: 1 ⊗ 1 = 0 ⊕ 1 ⊕ 2
function partA(rng)
    println("=== Part A: 1 (x) 1 = 0 + 1 + 2 ===")
    a, b = randn(rng, 3), randn(rng, 3)
    U0, U1, U2 = cg11(a, b)
    S = (a * b' + b * a') / 2 - dot(a, b) / 3 * I
    @printf("  L=0: U0 = %+.6f = a.b/sqrt(3);  L=1: max|U1 - (a x b)/sqrt(2)| = %.1e\n", U0, maximum(abs.(U1 - cross(a, b) / sqrt(2))))
    @printf("  L=2: |U2| = %.6f;  ||traceless symmetric part of a b'||_F = %.6f\n", norm(U2), norm(S))
    @printf("  |a|^2|b|^2 = %.6f = sum_L |U_L|^2 = %.6f\n", norm(a)^2 * norm(b)^2, U0^2 + norm(U1)^2 + norm(U2)^2)
    R = rand_rotation(rng)
    V0, V1, V2 = cg11(R * a, R * b)
    @printf("  intertwiner: L=0 %.1e, L=1 %.1e, L=2 %.1e   (|U_L(Ra, Rb) - D^L(R) U_L(a, b)|)\n",
            abs(V0 - U0), maximum(abs.(V1 - R * U1)), maximum(abs.(V2 - wigner_D(2, R) * U2)))
end

# ---------------------------------------------------------------- Part B: 점 합성곱 + 게이트
const M0, M1 = 8, 8                   # 입력 다중도 0e, 1o
const NS, NG, N1, N2 = 8, 16, 8, 8    # 출력: 스칼라 8 + 게이트 16 (0e 24개), 1o 8, 2e 8
# 경로 (이름, 입력 다중도, 출력 다중도): 0e⊗0e→0e, 0e⊗1o→1o, 0e⊗2e→2e, 1o⊗0e→1o, 1o⊗1o→0e, 1o⊗1o→2e, 1o⊗2e→1o
const PATHS = [(:ss, M0, NS + NG), (:sv, M0, N1), (:st, M0, N2), (:vs, M1, N1), (:vv0, M1, NS + NG), (:vv2, M1, N2), (:vt, M1, N1)]
const NW = sum(p[2] * p[3] for p in PATHS)                                 # 간선당 가중치 704

soft_unit_step(x) = x > 0 ? exp(-1 / x) : zero(x)
"e3nn soft_one_hot_linspace(basis = 'smooth_finite', cutoff = True)"
function smooth_finite(d, rc, nb)
    T = eltype(d)
    vals = T.(collect(range(0, rc; length = nb + 2))[2:end-1]); step = vals[2] - vals[1]
    df = (reshape(d, 1, :) .- vals) ./ step
    return T(1.14136 * exp(2)) .* soft_unit_step.(df .+ 1) .* soft_unit_step.(1 .- df)
end

"반경 MLP 8 → 16 (SiLU) → 704, PyTorch Linear식 초기화"
function radial_params(rng; nb = 8, nh = 16)
    u(o, i) = (rand(rng, o, i) .* 2 .- 1) ./ sqrt(i)
    return (W1 = u(nh, nb), b1 = vec(u(nh, nb)[:, 1]), W2 = u(NW, nh), b2 = vec(u(NW, nh)[:, 1]))
end

"h = (s: 8 x N, v: 3 x 8 x N), pos: 3 x N → (s: 8, v: 3 x 8, t: 5 x 8) per atom. 메시지는 src → dst, vec = pos[src] - pos[dst]"
function pointconv(h, pos, P; rc = 4.0)
    N = size(pos, 2)
    src = [j for i in 1:N for j in 1:N if i != j]; dst = [i for i in 1:N for j in 1:N if i != j]
    vec_ = pos[:, src] .- pos[:, dst]; E = length(src)
    Y0, Y1, Y2 = sh_batch(vec_)
    emb = smooth_finite(vec(sqrt.(sum(abs2, vec_; dims = 1))), rc, 8)
    w = P.W2 * NNlib.swish.(P.W1 * emb .+ P.b1) .+ P.b2                   # 704 x E
    off = 0; Wp = Dict{Symbol,Array{eltype(pos),3}}()
    for (nm, a, c) in PATHS
        Wp[nm] = reshape(w[off+1:off+a*c, :], a, c, E); off += a * c
    end
    s, v = h.s[:, src], h.v[:, :, src]                                     # 8 x E, 3 x 8 x E
    mix(x, W) = batched_mul(x, W)                                          # (k, a, E) x (a, c, E) -> (k, c, E)
    o0 = mix(reshape(s .* Y0, 1, M0, E), Wp[:ss]) .+
         mix(reshape(sum(v .* reshape(Y1, 3, 1, E); dims = 1) ./ sqrt(eltype(pos)(3)), 1, M1, E), Wp[:vv0])        # 1o·1o → 0e
    o1 = mix(reshape(Y1, 3, 1, E) .* reshape(s, 1, M0, E), Wp[:sv]) .+ mix(v .* reshape(Y0, 1, 1, E), Wp[:vs]) .+
         mix(batched_mul(m2_batch(Y2), v), Wp[:vt])                                                     # 1o ⊗ 2e → 1o: M(Y₂) v
    o2 = mix(reshape(Y2, 5, 1, E) .* reshape(s, 1, M0, E), Wp[:st]) .+
         mix(u2_outer(v, repeat(reshape(Y1, 3, 1, E), 1, M1, 1)), Wp[:vv2])                             # 1o ⊗ 1o → 2e
    agg(x) = NNlib.scatter(+, x, dst; dstsize = (size(x)[1:end-1]..., N))
    a0, a1, a2 = agg(dropdims(o0; dims = 1)), agg(o1), agg(o2)            # 24 x N, 3 x 8 x N, 5 x 8 x N
    g = NNlib.sigmoid.(a0[NS+1:end, :])                                    # 게이트: 1o 8개, 2e 8개 순서
    return (s = NNlib.swish.(a0[1:NS, :]), v = a1 .* reshape(g[1:N1, :], 1, N1, N), t = a2 .* reshape(g[N1+1:end, :], 1, N2, N))
end

"O(3) 원소 Q를 특징에: 0e 그대로, 1o는 Q, 2e는 D²(Q) (= D²(det Q · Q))"
act_in(h, Q) = (s = h.s, v = reshape(Q * reshape(h.v, 3, :), size(h.v)))
act_out(o, Q) = (s = o.s, v = reshape(Q * reshape(o.v, 3, :), size(o.v)), t = reshape(irrep_D(2, 1, Q) * reshape(o.t, 5, :), size(o.t)))
maxdiff(a, b) = maximum(maximum(abs.(getfield(a, k) .- getfield(b, k))) for k in keys(a))

function partB(rng, pos64)
    println("=== Part B: equivariant point convolution + gate on an ethanol geometry ===")
    for T in (Float32, Float64)
        P = map(x -> T.(x), radial_params(Xoshiro(0))); pos = T.(pos64)
        h = (s = randn(rng, T, M0, 9), v = randn(rng, T, 3, M1, 9))
        out = pointconv(h, pos, P)
        er, ei = 0.0, 0.0
        for _ in 1:20
            R = T.(rand_rotation(rng))
            for (Q, improper) in ((R, false), (-R, true))
                e = maxdiff(pointconv(act_in(h, Q), Q * pos, P), act_out(out, Q))
                improper ? (ei = max(ei, e)) : (er = max(er, e))
            end
        end
        t = randn(rng, T, 3); perm = randperm(rng, 9)
        etr = maxdiff(pointconv(h, pos .+ t, P), out)
        hp = (s = h.s[:, perm], v = h.v[:, :, perm])
        op = pointconv(hp, pos[:, perm], P)
        epm = maxdiff(op, (s = out.s[:, perm], v = out.v[:, :, perm], t = out.t[:, :, perm]))
        @printf("  %s: out 8x0e+8x1o+8x2e (dim %d); max equivariance error: SO(3) %.1e, improper O(3) %.1e, translation %.1e, permutation %.1e\n",
                T, 8 + 24 + 40, er, ei, etr, epm)
    end
end

function partC(rng)
    println("=== Part C: which nonlinearity keeps equivariance? (l=1 block, 1000 random rotations) ===")
    v = randn(rng, 3, 8); en, eg = 0.0, 0.0
    naive(x) = NNlib.swish.(x)
    normgate(x) = NNlib.sigmoid.(sqrt.(sum(abs2, x; dims = 1))) .* x
    for _ in 1:1000
        R = rand_rotation(rng)
        en = max(en, maximum(abs.(naive(R * v) - R * naive(v)))); eg = max(eg, maximum(abs.(normgate(R * v) - R * normgate(v))))
    end
    @printf("  component-wise SiLU on vectors: %.2f (|v|_max %.2f);  norm gate sigma(|v|) v: %.1e\n", en, maximum(abs.(v)), eg)
end

function partD()
    println("=== Part D: parameter counts ===")
    nrad = 8 * 16 + 16 + 16 * NW + NW
    @printf("  TP paths = %d (e3nn counts 9: the 0e outputs are split into 8 scalars + 16 gates), weights per edge = %d, radial MLP params = %d\n",
            length(PATHS), NW, nrad)
    din, dsh, dout = M0 + 3M1, 1 + 3 + 5, 24 + 3N1 + 5N2
    @printf("  unconstrained bilinear map (%d) x (%d) -> %d: %d entries = %.1f x the TP weights\n", din, dsh, dout, din * dsh * dout, din * dsh * dout / NW)
    d16 = 16 + 48 + 80
    @printf("  equivariant linear on 16x0e+16x1o+16x2e (dim %d): %d weights; dense %d->%d: %d (%.1fx)\n", d16, 3 * 16^2, d16, d16, d16^2 + d16, (d16^2 + d16) / (3 * 16^2))
end

function main()
    rng = Xoshiro(0)
    d = npzread(joinpath(@__DIR__, "..", "data", "ethanol_xtb_300K.npz"))
    pos = permutedims(d["R"][1, :, :])
    partA(rng); partB(rng, pos); partC(rng); partD()
end

main()

# ---------------------------------------------------------------- e3nn을 그대로 부르는 패턴 (PythonCall)
# using PythonCall
# torch = pyimport("torch"); o3 = pyimport("e3nn.o3"); torch.set_default_dtype(torch.float64)
# tp = o3.FullyConnectedTensorProduct("8x0e + 8x1o", o3.Irreps.spherical_harmonics(2), "24x0e + 8x1o + 8x2e"; shared_weights = false)
# println(pyconvert(Int, tp.weight_numel))                        # 704: 위 NW와 같다
# x = torch.tensor(randn(9 * 40, 32)); y = torch.tensor(randn(9 * 40, 9)); w = torch.tensor(randn(9 * 40, 704))
# out = pyconvert(Matrix{Float64}, tp(x, y, w).detach().numpy())  # (간선, 88). 미분까지 e3nn에 맡기려면 학습 루프 전체를 파이썬에 둔다
