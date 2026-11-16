# descriptors.jl — Coulomb 행렬, sine 행렬, ACSF(G1, G2, G4)를 Julia로 직접 구현 (SOAP·MBTR은 dscribe_bridge.jl)
# 배치 약속(column-major): 좌표 R은 3 x N(열 = 원자), 셀 B는 3 x 3(열 = 격자 벡터).
#   파이썬(ASE)의 positions(N x 3), cell(행 = 격자 벡터)과는 전치 관계다.
# ACSF는 DScribe 2.1의 정의·순서를 그대로 따른다(DScribe 출력과 기계 정밀도로 일치함을 확인했다):
#   종(species)은 원자번호 오름차순, 종마다 [G1, G2...], 그다음 종 쌍 (a <= b)마다 G4 블록.
#   G4는 이웃 쌍 j < k에 대해 r_jk < r_c인 것만 더하고 fc(r_jk)를 곱한다.
using LinearAlgebra

# ---------------------------------------------------------------- 원자 구조
"원자 구조: Z(원자번호), R(3 x N, Å), cell(3 x 3 열 = 격자 벡터, 비주기면 nothing)"
struct Structure
    Z::Vector{Int}
    R::Matrix{Float64}
    cell::Union{Nothing,Matrix{Float64}}
end
natoms(s::Structure) = length(s.Z)

"fcc 관습 세포(4원자)를 rep x rep x rep로 반복한 초세포"
function fcc_supercell(Z, a, rep)
    basis = ([0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0])   # ASE bulk(cubic=True)와 같은 순서
    R = Float64[]
    for i in 0:rep-1, j in 0:rep-1, k in 0:rep-1, b in basis
        append!(R, a .* ([i, j, k] .+ b))
    end
    R = reshape(R, 3, :)
    return Structure(fill(Z, size(R, 2)), R, Matrix(a * rep * I(3)))
end

"모든 원자 좌표를 셀 안으로 접는다 (ASE wrap)"
function wrap(s::Structure)
    F = s.cell \ s.R
    return Structure(s.Z, s.cell * (F .- floor.(F)), s.cell)
end

# ---------------------------------------------------------------- Coulomb 행렬과 sine 행렬
"Coulomb 행렬: M_ij = Z_i Z_j / |R_i - R_j|, M_ii = 0.5 Z_i^2.4"
function coulomb_matrix(Z, R)
    N = length(Z); M = zeros(N, N)
    for j in 1:N, i in 1:N
        M[i, j] = i == j ? 0.5 * Z[i]^2.4 : Z[i] * Z[j] / norm(R[:, i] - R[:, j])
    end
    return M
end

"sine 행렬(Faber 등 2015): 분수 좌표 차이 f = B \\ (R_i - R_j), 거리 |B * sin^2(pi f)|"
function sine_matrix(Z, R, B)
    N = length(Z); M = zeros(N, N)
    for j in 1:N, i in 1:N
        if i == j
            M[i, j] = 0.5 * Z[i]^2.4
        else
            f = B \ (R[:, i] - R[:, j])
            M[i, j] = Z[i] * Z[j] / norm(B * sin.(pi .* f) .^ 2)
        end
    end
    return M
end
sine_matrix(s::Structure) = sine_matrix(s.Z, s.R, s.cell)

"행 노름 내림차순으로 행·열을 함께 정렬 (sortperm은 안정 정렬 = numpy kind='stable')"
function sorted_rows(M)
    p = sortperm(-[norm(r) for r in eachrow(M)])
    return M[p, p]
end
eigs(M) = sort(eigvals(Symmetric(M)); rev = true)

# ---------------------------------------------------------------- ACSF
const G2_PARAMS = [(eta, 0.0) for eta in (0.01, 0.05, 0.1, 0.3, 0.6, 1.0, 2.0, 4.0)]
const G4_PARAMS = [(0.005, z, lam) for z in (1, 2, 4, 16) for lam in (-1, 1)]
fcut(r, rc) = r < rc ? 0.5 * (cos(pi * r / rc) + 1) : 0.0

"원자 i의 이웃 (j, 변위 벡터). 주기계면 r_c를 덮는 셀 이미지를 모두 훑는다."
function neighbor_vectors(s::Structure, i, rc)
    out = Tuple{Int,Vector{Float64}}[]; N = natoms(s)
    if s.cell === nothing
        for j in 1:N
            j == i && continue
            v = s.R[:, j] - s.R[:, i]
            norm(v) < rc && push!(out, (j, v))
        end
    else
        a1, a2, a3 = eachcol(s.cell); V = abs(det(s.cell))
        w = (V / norm(cross(a2, a3)), V / norm(cross(a3, a1)), V / norm(cross(a1, a2)))   # 면 사이 거리
        nr = ceil.(Int, rc ./ w)
        for n1 in -nr[1]:nr[1], n2 in -nr[2]:nr[2], n3 in -nr[3]:nr[3]
            sh = s.cell * [n1, n2, n3]
            for j in 1:N
                (j == i && n1 == 0 && n2 == 0 && n3 == 0) && continue
                v = s.R[:, j] + sh - s.R[:, i]
                norm(v) < rc && push!(out, (j, v))
            end
        end
    end
    return out
end

"원자 i의 ACSF 벡터. species는 원자번호 목록(오름차순으로 정렬해 쓴다)."
function acsf(s::Structure, i; species, rc = 5.0, g2 = G2_PARAMS, g4 = G4_PARAMS)
    sp = sort(species); S = length(sp); sidx = Dict(z => k for (k, z) in enumerate(sp))
    pairs = [(a, b) for a in 1:S for b in a:S]; pidx = Dict(p => k for (k, p) in enumerate(pairs))
    nb = neighbor_vectors(s, i, rc)
    rad = zeros(1 + length(g2), S)                    # 열 = 종 -> vec 하면 종별 [G1, G2...] 순서
    ang = zeros(length(g4), length(pairs))
    for (j, v) in nb
        r = norm(v); k = sidx[s.Z[j]]; f = fcut(r, rc)
        rad[1, k] += f
        for (q, (eta, rs)) in enumerate(g2)
            rad[1+q, k] += exp(-eta * (r - rs)^2) * f
        end
    end
    if !isempty(g4)
        for a in 1:length(nb), b in a+1:length(nb)
            j, vj = nb[a]; k, vk = nb[b]
            rij = norm(vj); rik = norm(vk); rjk = norm(vk - vj)
            rjk < rc || continue
            c = dot(vj, vk) / (rij * rik)
            p = pidx[minmax(sidx[s.Z[j]], sidx[s.Z[k]])]
            f3 = fcut(rij, rc) * fcut(rik, rc) * fcut(rjk, rc); e2 = rij^2 + rik^2 + rjk^2
            for (q, (eta, zeta, lam)) in enumerate(g4)
                ang[q, p] += 2.0^(1 - zeta) * (1 + lam * c)^zeta * exp(-eta * e2) * f3
            end
        end
    end
    return vcat(vec(rad), vec(ang))
end

"모든 원자의 ACSF: N x dim (DScribe create와 같은 모양)"
acsf_all(s::Structure; kw...) = permutedims(reduce(hcat, [acsf(s, i; kw...) for i in 1:natoms(s)]))
