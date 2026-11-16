# lab_7_12.jl — Lab 7.12: 등변성과 보존성을 분리해서 재기 (labs/73a_equiv.py)
# 네 힘 모형(Float64): (a) 학습된 SchNet 에너지 + 자동미분 힘, (b) 원좌표 MLP 에너지 + 자동미분 힘,
# (c) 원좌표 MLP가 힘 27성분을 직접 출력, (d) SchNet 특징 + 직접 등변 머리 F_i = Σ_j s_ij r̂_ij.
# (a)의 가중치는 파이썬이 저장한 72b_schnet_ethanol_N1000.pt를 PythonCall로 옮기거나(load_torch_schnet!),
# lab_7_9.jl처럼 Julia에서 학습한 ps를 쓴다. 힘 야코비안은 에너지 모형이면 ForwardDiff.hessian(정확히 대칭),
# 직접 힘 모형이면 ForwardDiff.jacobian으로 구한다.
include("ch7_common.jl")
include("mlip_common.jl")
using PythonCall

struct XYZEnergy{C} <: Lux.AbstractLuxContainerLayer{(:f,)}
    f::C
end
(m::XYZEnergy)(R, ps, st) = (vec(first(m.f(reshape(R .- mean(R; dims = 2), 27, :), ps.f, st.f))), st)
struct XYZForce{C} <: Lux.AbstractLuxContainerLayer{(:f,)}
    f::C
end
(m::XYZForce)(R, ps, st) = (reshape(first(m.f(reshape(R .- mean(R; dims = 2), 27, :), ps.f, st.f)), 3, 9, :), st)

"직접 등변 힘: s_ij = MLP([h_i + h_j; rbf(r_ij)])는 (i, j) 대칭·불변, F_i = Σ_j s_ij (R_j - R_i)/r_ij"
struct DirectEquivariant{N,P} <: Lux.AbstractLuxContainerLayer{(:net, :pair)}
    net::N
    pair::P
end
function (m::DirectEquivariant)(R, ps, st)
    h = atom_features(m.net, R, ps.net, st.net)                           # F x n x B
    F_, n, B = size(h)
    vec_, r, offd = pair_geometry(R)
    rbf = gaussian_rbf(r, m.net.mu, m.net.gamma)
    hij = reshape(reshape(h, F_, n, 1, B) .+ reshape(h, F_, 1, n, B), F_, :)
    s, _ = m.pair(vcat(hij, rbf), ps.pair, st.pair)
    s = reshape(s, 1, n, n, B) .* reshape(offd, 1, n, n, 1)
    return dropdims(sum(s .* vec_ ./ reshape(r, 1, n, n, B); dims = 3); dims = 3), st
end

function main()
    R, Z, E, F = load_ethanol(300)
    _, _, te = split_300K()
    X = R[:, :, te[1:20]]
    rng = Xoshiro(1)
    schnet = DenseSchNet(Z .+ 1; nf = 32, nrbf = 20, rc = 5.0, ninter = 3)
    psA, stA = Lux.setup(Xoshiro(0), schnet)
    load_torch_schnet!(psA, joinpath(LABS, "72b_schnet_ethanol_N1000.pt"); model = schnet); schnet.e0[] = mean(E[1:2000])
    mB = XYZEnergy(Chain(ldense(27, 64, swish), ldense(64, 64, swish), ldense(64, 1)))
    mC = XYZForce(Chain(ldense(27, 64, swish), ldense(64, 64, swish), ldense(64, 27)))
    mD = DirectEquivariant(DenseSchNet(Z .+ 1; nf = 32, nrbf = 20, rc = 5.0, ninter = 2), Chain(ldense(52, 32, swish), ldense(32, 1)))
    models = ["(a) SchNet E -> -dE/dR" => (:E, schnet, psA, stA), "(b) xyz-MLP E -> -dE/dR" => (:E, mB, Lux.setup(Xoshiro(0), mB)...),
              "(c) xyz-MLP direct F" => (:F, mC, Lux.setup(Xoshiro(0), mC)...), "(d) SchNet direct equiv. F" => (:F, mD, Lux.setup(Xoshiro(0), mD)...)]
    Qs = random_rotations(rng, 200); Qs[:, :, 101:200] .*= -1                 # 뒤의 절반: 부적절(det = -1)
    ts = 5.0 .* randn(rng, 3, 200)
    @printf("%-28s %11s %19s %10s %9s %12s %11s\n", "model", "E inv. err", "F equiv. err (SO3)", "(det -1)", "transl.", "asym(dF/dR)", "loop work")
    summ = []
    for (name, (kind, m, ps, st)) in models
        ps = to64(ps)
        Efun(Y) = first(m(Y, ps, st))
        forces(Y) = kind == :E ? (Efun(Y), -only(Zygote.gradient(y -> sum(Efun(y)), Y))) : (nothing, first(m(Y, ps, st)))
        E0, F0 = forces(X); scale = mean(abs.(F0))
        eE, eF = Float64[], Float64[]
        for k in 1:200
            Xg = rotate_batch(Qs[:, :, k:k] .* ones(1, 1, 20), X) .+ ts[:, k]
            Eg, Fg = forces(Xg)
            push!(eF, maximum(abs.(Fg .- rotate_batch(Qs[:, :, k:k] .* ones(1, 1, 20), F0))) / scale)
            Eg === nothing || push!(eE, maximum(abs.(Eg .- E0)))
        end
        _, Ft = forces(X .+ [3.0, -2.0, 1.0])
        x0 = vec(X[:, :, 1])
        J = kind == :E ? -ForwardDiff.hessian(v -> only(Efun(reshape(v, 3, 9, 1))), x0) :
                         ForwardDiff.jacobian(v -> vec(forces(reshape(v, 3, 9, 1))[2]), x0)
        asym = norm(J - J') / norm(J + J')
        θ = range(0, 2pi; length = 401)                                     # 원자 4(수산기 H)를 xy 평면의 반지름 0.1 Å 원으로
        path = repeat(X[:, :, 1:1], 1, 1, 401)
        path[1, 4, :] .+= 0.1 .* cos.(θ); path[2, 4, :] .+= 0.1 .* sin.(θ)
        _, Fp = forces(path)
        dr = path[:, 4, 2:end] .- path[:, 4, 1:end-1]; Fm = (Fp[:, 4, 2:end] .+ Fp[:, 4, 1:end-1]) ./ 2
        Wloop = sum(Fm .* dr)
        Xc = X .- mean(X; dims = 2)
        netF = maximum(norm.(eachcol(dropdims(sum(F0; dims = 2); dims = 2)))) / scale
        torque = maximum(norm(sum(cross(Xc[:, i, b], F0[:, i, b]) for i in 1:9)) for b in 1:20) / scale
        @printf("%-28s %11.1e %19.1e %10.1e %9.1e %12.1e %10.2e meV\n", name, isempty(eE) ? NaN : maximum(eE),
                maximum(eF[1:100]), maximum(eF[101:200]), maximum(abs.(Ft .- F0)) / scale, asym, Wloop * 1000)
        push!(summ, (name, netF, torque))
    end
    @printf("%-28s %18s %28s\n", "model", "|sum_i F_i|/<|F|>", "|sum_i r_i x F_i|/<|F|> (A)")
    for (name, a, b) in summ
        @printf("%-28s %18.1e %28.1e\n", name, a, b)
    end
end

main()
