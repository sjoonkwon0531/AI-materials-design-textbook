# lab_7_19.jl — Lab 7.19: NequIP형 상호작용 블록으로 만든 에너지 모형의 대칭 점검 (labs/76_e3nn_block.py)
# equiv_layers.jl의 손으로 쓴 등변 블록(16x0e + 8x1o + 8x1e, l ≤ 1)을 두 층 쌓은 EquivPot으로
# 은닉 특징의 회전 등변성, 에너지 불변성, 힘 등변성, 반전(1o 대 1e), 이동, 순열, 매끄러운 차단, '스칼라로 잘못 표시한
# 방향' 버그, Float32 오차를 잰다. 학습하지 않은 모형(Float64)으로 충분하다.
include("ch7_common.jl")
include("mlip_common.jl")
include("so3.jl")
include("equiv_layers.jl")

function main()
    R, Z, _, _ = load_ethanol(300)
    pos = R[:, :, 1:4]                                                     # 분자 4개
    model = EquivPot(Z)
    ps, st = Lux.setup(Xoshiro(0), model); ps = to64(ps)
    blk = model.blocks.layers[2]
    @printf("block 2: TP paths = %d, weights per edge = %d, block params = %d; model params = %d\n",
            length(block_paths(blk)), weight_numel(blk), Lux.parameterlength(ps.blocks.layer_2), Lux.parameterlength(ps))
    EF(X) = (E = first(model(X, ps, st)); (E, -only(Zygote.gradient(x -> sum(first(model(x, ps, st))), X))))
    Q = rand_rotation(Xoshiro(1))
    rotx(Q, X) = reshape(Q * reshape(X, 3, :), size(X))
    E0, F0 = EF(pos); E1, F1 = EF(rotx(Q, pos))
    h0 = node_features(model, pos, ps, st)[2]; h1 = node_features(model, rotx(Q, pos), ps, st)[2]
    eh = max(maximum(abs.(h1.s - h0.s)), maximum(abs.(h1.v - rotx(Q, h0.v))), maximum(abs.(h1.p - rotx(det(Q) * Q, h0.p))))
    @printf("hidden features of block 2: max |h(Rx) - D(R) h(x)| = %.1e (max |h| = %.2f)\n", eh, maximum(abs.(h0.s)))
    @printf("energy: max |E(Rx) - E(x)| = %.1e eV;  forces: max |F(Rx) - R F(x)| = %.1e eV/A\n", maximum(abs.(E1 - E0)), maximum(abs.(F1 - rotx(Q, F0))))
    P = -Matrix(1.0I, 3, 3)
    E2, F2 = EF(rotx(P, pos)); h2 = node_features(model, rotx(P, pos), ps, st)[2]
    ehp = max(maximum(abs.(h2.s - h0.s)), maximum(abs.(h2.v + h0.v)), maximum(abs.(h2.p - h0.p)))   # 1o는 부호가 바뀌고 1e는 그대로
    @printf("inversion: max |h(-x) - D(P) h(x)| = %.1e, |E(-x) - E(x)| = %.1e, |F(-x) + F(x)| = %.1e\n", ehp, maximum(abs.(E2 - E0)), maximum(abs.(F2 + F0)))
    E3, F3 = EF(pos .+ [1.3, -0.7, 2.1])
    @printf("translation: |dE| = %.1e, |dF| = %.1e\n", maximum(abs.(E3 - E0)), maximum(abs.(F3 - F0)))
    perm = [1, 2, 3, 4, 5, 6, 8, 7, 9]                                    # 분자 0의 메틸 H 두 개 교환
    pp = copy(pos); pp[:, :, 1] = pos[:, perm, 1]
    E4, F4 = EF(pp)
    @printf("permutation of two H: |dE| = %.1e, |F_perm - P F| = %.1e\n", maximum(abs.(E4 - E0)), maximum(abs.(F4[:, :, 1] - F0[:, perm, 1])))
    println("Bessel x envelope (n = 1) at r = 4.5, 4.9, 4.99, 5.0 A: ",
            [@sprintf("%.1e", v) for v in bessel_basis([4.5, 4.9, 4.99, 5.0], 5.0, 8)[1, :]])
    bad = EquivPot(Z; bug = true); pb, sb = Lux.setup(Xoshiro(0), bad); pb = to64(pb)
    EFb(X) = (first(bad(X, pb, sb)), -only(Zygote.gradient(x -> sum(first(bad(x, pb, sb))), X)))
    Eb0, Fb0 = EFb(pos); Eb1, Fb1 = EFb(rotx(Q, pos))
    @printf("bug (xyz labelled 4x0e): |E(Rx) - E(x)| = %.1e eV (spread of E over 4 frames %.1e), |F(Rx) - R F(x)|/|F| = %.2f\n",
            maximum(abs.(Eb1 - Eb0)), maximum(Eb0) - minimum(Eb0), norm(Fb1 - rotx(Q, Fb0)) / norm(Fb0))
    ps32 = Functors.fmap(x -> x isa AbstractArray{Float64} ? Float32.(x) : x, ps)
    Ea = first(model(Float32.(pos), ps32, st)); Eb = first(model(Float32.(rotx(Q, pos)), ps32, st))
    @printf("float32: |E(Rx) - E(x)| = %.1e\n", maximum(abs.(Eb - Ea)))
end

main()
