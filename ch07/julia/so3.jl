# so3.jl — 등변성 Lab(7.11–7.15, 7.19)의 손으로 쓴 SO(3)/O(3) 도구
# 실수 구면조화함수 l ≤ 2, Wigner D(l = 1, 2), 1 ⊗ 1 = 0 ⊕ 1 ⊕ 2 Clebsch–Gordan, 르장드르 덧셈 정리, 회전 생성.
# 기저 약속: l = 1은 Cartesian (x, y, z) 순서(Y₁ = √3 r̂, D¹(R) = R; e3nn과 같다).
#            l = 2는 트레이스 없는 대칭 행렬 S의 정규직교 좌표 u2(S) = (√2 S_xy, √2 S_yz, √(3/2) S_zz, √2 S_xz, (S_xx - S_yy)/√2).
#            Y₂(r̂) = √(15/2) u2(r̂ r̂ᵀ)이고 정규화는 e3nn의 'component'(|Y_l|² = 2l + 1)와 같다.
#            e3nn의 l = 2 성분은 축 이름을 바꾼 같은 함수다: Y₂^e3nn(x, y, z) = Y₂^여기(z, x, y).
using LinearAlgebra, Random

const SQ2 = sqrt(2.0)

"대칭 3x3 행렬 M의 트레이스 없는 부분 → 5성분 (l = 2 좌표)"
function u2(M::AbstractMatrix)
    T = float(eltype(M)); s2, s15 = sqrt(T(2)), sqrt(T(1.5))         # Float32 입력은 Float32로 계산
    S = (M + transpose(M)) / 2
    t = tr(S) / 3
    return [s2 * S[1, 2], s2 * S[2, 3], s15 * (S[3, 3] - t), s2 * S[1, 3], (S[1, 1] - S[2, 2]) / s2]
end
"u2의 역: 5성분 → 트레이스 없는 대칭 행렬"
function m2(t::AbstractVector)
    T = float(eltype(t)); s2 = sqrt(T(2))
    zz = t[3] / sqrt(T(1.5))
    xx, yy = (-zz + s2 * t[5]) / 2, (-zz - s2 * t[5]) / 2
    a, b, d = t[1] / s2, t[2] / s2, t[4] / s2
    return [xx a d; a yy b; d b zz]
end

"실수 구면조화 Y_l(r̂), l = 0, 1, 2 (component 정규화)"
function sh(l::Int, r::AbstractVector)
    u = r / norm(r)
    l == 0 && return [one(eltype(u))]
    T = eltype(u)
    l == 1 && return sqrt(T(3)) .* u
    l == 2 && return sqrt(T(7.5)) .* u2(u * transpose(u))
    error("l ≤ 2만 손으로 썼다. 더 큰 l의 불변량은 legendre_power, 성분은 PythonCall로 e3nn을 부른다")
end
"열마다 방향 하나(3 x E) → (Y₀ 1 x E, Y₁ 3 x E, Y₂ 5 x E). 신경망 층용 (Zygote 친화적: 배열 변경 없음)"
function sh_batch(V::AbstractMatrix)
    T = eltype(V); s2 = sqrt(T(2))
    u = V ./ sqrt.(sum(abs2, V; dims = 1))
    x, y, z = u[1:1, :], u[2:2, :], u[3:3, :]
    Y2 = sqrt(T(7.5)) .* vcat(s2 .* x .* y, s2 .* y .* z, sqrt(T(1.5)) .* (z .^ 2 .- T(1 / 3)), s2 .* x .* z, (x .^ 2 .- y .^ 2) ./ s2)
    return one.(x), sqrt(T(3)) .* u, Y2
end

"Wigner D^l(R) (고유 회전 R). D¹ = R, D²의 k번째 열 = u2(R m2(e_k) Rᵀ)"
function wigner_D(l::Int, R::AbstractMatrix{T}) where {T}
    l == 0 && return ones(T, 1, 1)
    l == 1 && return Matrix(R)
    l == 2 && return reduce(hcat, [u2(R * m2(T.(1:5 .== k)) * transpose(R)) for k in 1:5])
    error("l ≤ 2")
end
"O(3) 원소 Q에 대한 irrep (l, p)의 표현: Q = det(Q) R로 쓰고 p^{[det < 0]} D^l(R)"
function irrep_D(l::Int, p::Int, Q::AbstractMatrix)
    s = det(Q) < 0 ? -1 : 1
    return (s < 0 && p < 0 ? -1 : 1) .* wigner_D(l, s .* Q)
end

"1 ⊗ 1 = 0 ⊕ 1 ⊕ 2 (정규직교 기저 변환): (a·b/√3, (a × b)/√2, u2(a bᵀ)). Σ_L |U_L|² = |a|²|b|²"
cg11(a, b) = (dot(a, b) / sqrt(3), cross(a, b) / SQ2, u2(a * transpose(b)))
"u2(sym(a bᵀ))를 배열 a, b (3 x …)에 대해 원소별로 (층 안의 1o ⊗ 1o → 2e 경로)"
function u2_outer(a, b)
    a1, a2, a3 = a[1:1, ntuple(_ -> :, ndims(a) - 1)...], a[2:2, ntuple(_ -> :, ndims(a) - 1)...], a[3:3, ntuple(_ -> :, ndims(a) - 1)...]
    b1, b2, b3 = b[1:1, ntuple(_ -> :, ndims(b) - 1)...], b[2:2, ntuple(_ -> :, ndims(b) - 1)...], b[3:3, ntuple(_ -> :, ndims(b) - 1)...]
    T = eltype(a); s2 = sqrt(T(2))
    t = (a1 .* b1 .+ a2 .* b2 .+ a3 .* b3) ./ 3
    return vcat(s2 .* (a1 .* b2 .+ a2 .* b1) ./ 2, s2 .* (a2 .* b3 .+ a3 .* b2) ./ 2, sqrt(T(1.5)) .* (a3 .* b3 .- t),
                s2 .* (a1 .* b3 .+ a3 .* b1) ./ 2, (a1 .* b1 .- a2 .* b2) ./ s2)
end
"5 x E 배열의 각 열을 3 x 3 트레이스 없는 대칭 행렬로 (3 x 3 x E) — 1o ⊗ 2e → 1o 경로 M(t) v에 쓴다"
function m2_batch(T5::AbstractMatrix)
    T = eltype(T5); s2 = sqrt(T(2))
    zz = T5[3:3, :] ./ sqrt(T(1.5))
    xx, yy = (-zz .+ s2 .* T5[5:5, :]) ./ 2, (-zz .- s2 .* T5[5:5, :]) ./ 2
    a, b, d = T5[1:1, :] ./ s2, T5[2:2, :] ./ s2, T5[4:4, :] ./ s2
    return reshape(vcat(xx, a, d, a, yy, b, d, b, zz), 3, 3, :)
end

"르장드르 다항식 P_l(x)"
function legendre(l, x)
    l == 0 && return one(x)
    p0, p1 = one(x), x
    for k in 2:l
        p0, p1 = p1, ((2k - 1) * x * p1 - (k - 1) * p0) / k
    end
    return p1
end
"덧셈 정리: |Σ_j f_j Y_l(r̂_j)|² = (2l + 1) Σ_jk f_j f_k P_l(r̂_j · r̂_k). U는 3 x k 단위벡터 열 (모든 l에 성립)"
legendre_power(l, U, f) = (2l + 1) * dot(f, legendre.(l, transpose(U) * U) * f)

"균일 무작위 고유 회전"
function rand_rotation(rng = Random.default_rng())
    F = qr(randn(rng, 3, 3)); Q = Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
    det(Q) < 0 && (Q[:, 1] .*= -1)
    return Q
end
"Rodrigues: 축 a 둘레로 각 θ"
function rot(axis, θ)
    a = axis / norm(axis)
    K = [0 -a[3] a[2]; a[3] 0 -a[1]; -a[2] a[1] 0]
    return I + sin(θ) * K + (1 - cos(θ)) * K * K
end
"법선 n의 거울"
mirror(n) = (u = n / norm(n); I - 2 * u * transpose(u))
