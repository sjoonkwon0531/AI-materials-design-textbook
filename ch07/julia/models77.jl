# models77.jl — Lab 7.21–7.22의 모형 정의: DirectMLP(원좌표 / 분자 좌표계), 사원수 무작위 회전, 종류별 생성
# mlip_common.jl(DenseSchNet)과 equiv_layers.jl(EquivPot, crossb) 다음에 include한다.
"frame = false: 중심화 좌표 27개(이동만 불변). frame = true: C1→C2, C2→O로 만든 Gram–Schmidt 좌표계의 성분(회전 불변)"
struct DirectMLP{C} <: Lux.AbstractLuxContainerLayer{(:net,)}
    net::C
    frame::Bool
    e0::Base.RefValue{Float64}
    scale::Base.RefValue{Float64}
end
DirectMLP(; h = 128, frame = false) = DirectMLP(Chain(ldense(27, h, swish), ldense(h, h, swish), ldense(h, h, swish), ldense(h, 1)), frame, Ref(0.0), Ref(1.0))
nrmz(v) = v ./ sqrt.(sum(abs2, v; dims = 1))
function (m::DirectMLP)(R, ps, st)
    x = R .- mean(R; dims = 2)
    if m.frame
        e1 = nrmz(R[:, 2, :] .- R[:, 1, :]); v = R[:, 3, :] .- R[:, 2, :]
        e2 = nrmz(v .- sum(v .* e1; dims = 1) .* e1); e3 = crossb(e1, e2)
        x = vcat(sum(x .* reshape(e1, 3, 1, :); dims = 1), sum(x .* reshape(e2, 3, 1, :); dims = 1), sum(x .* reshape(e3, 3, 1, :); dims = 1))
    end
    e, _ = m.net(reshape(x, 27, :), ps.net, st.net)
    return vec(e) .* Float32(m.scale[]) .+ Float32(m.e0[]), st
end

"균일 무작위 회전(단위 사원수)"
function quat_rotations(rng, n)
    Q = zeros(Float32, 3, 3, n)
    for k in 1:n
        q = randn(rng, 4); a, b, c, d = q ./ norm(q)
        Q[:, :, k] = [a^2+b^2-c^2-d^2 2(b*c-a*d) 2(b*d+a*c); 2(b*c+a*d) a^2-b^2+c^2-d^2 2(c*d-a*b); 2(b*d-a*c) 2(c*d+a*b) a^2-b^2-c^2+d^2]
    end
    return Q
end

function build(kind, Z)
    kind in (:mlp, :mlp_aug) && return DirectMLP()
    kind == :mlp_frame && return DirectMLP(; frame = true)
    sp = Dict(1 => 1, 6 => 2, 8 => 3)
    kind == :schnet && return DenseSchNet([sp[z] for z in Z]; nf = 64, nrbf = 32, rc = 5.0, ninter = 3, maxz = 3, gamma = 10.0)
    return EquivPot(Z)
end
