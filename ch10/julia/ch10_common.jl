# ch10_common.jl: 10장 Julia 포트의 공용 부분 (Lab 10.1–10.13이 함께 쓴다)
#
#   1. 경로, 단위, 시간 측정
#   2. PythonCall 다리: numpy 자료 읽기, ase.Atoms 변환, 사전학습 범용 MLIP(MACE-MP-0, CHGNet, SevenNet, MatterSim)와
#      ASE EMT, PySCF를 ASE 계산기로 부르기. 이 모델들은 Julia에 대응물이 없으므로 PythonCall로만 쓴다.
#   3. 주기 이웃 목록(무차별 탐색 + 이미지 이동), 배치 그래프
#   4. 학생 모델 세 가지(Lux.jl): InvNet(SchNet형, l = 0), EqNet(PaiNN형, l ≤ 1), SchNetS(Ch.7 72b의 SchNet)
#      에너지, 보존력, 응력(대칭 변형의 미분), 직접 힘 머리. 힘을 손실에 넣는 학습은 Lux의 중첩 AD를 쓴다.
#   5. 학습 루프(Optimisers.jl의 Adam, PyTorch와 같은 코사인 일정), 평가
#   6. MD: 속도 Verlet NVE, BAOAB Langevin, Maxwell–Boltzmann 초기 속도, 위원회의 힘 불일치 ε_F
#   7. 결정론적 도구: 선형 맞춤, 보간, Birch–Murnaghan(V^(-2/3)의 3차식으로 선형 최소제곱), 유한차분 응력과 Hessian,
#      진동수, 최원점 선택(FPS), 힘의 시간 자기상관, 무작위 회전
#
# 배열 약속(column-major): 좌표와 힘은 3 x n(열 = 원자), 배치는 3 x n x B. 셀 행렬 H는 3 x 3이고 "열"이 격자 벡터다
# (ASE의 cell은 행이 격자 벡터이므로 H = cell'). 이미지 이동은 H * S(S는 정수 3-벡터)이고, 변형은 r → (I + ε) r, H → (I + ε) H다.
# 정밀도: 학생 모델은 모두 Float64로 계산한다(파이썬의 SchNet 학생은 float32였다).
using LinearAlgebra, Random, Statistics, Printf, Serialization
using Lux, Zygote, ForwardDiff, NNlib
import Optimisers
using PythonCall
import JSON3

# ================================================================ 1. 경로, 단위, 시간
const CH10 = normpath(joinpath(@__DIR__, ".."))
const DATA = joinpath(CH10, "data")                    # 파이썬 Lab이 만든 npz, json과 MACE-MP-0 small 모델 파일
const JOUT = joinpath(@__DIR__, "models")              # Julia 판이 만든 자료와 학습한 매개변수
isdir(JOUT) || mkpath(JOUT)

const GPA = 160.21766208                               # eV/Å³ → GPa
const KB = 8.617333262e-5                              # eV/K
const AMU = 103.6427                                   # 1 amu = 103.6427 eV fs²/Å² (질량을 이 단위로 쓰면 시간은 fs)
const MSI = 28.0855                                    # Si 질량(amu)
const THZ = sqrt(1.602176634e-19 / 1.66053906660e-27) * 1e10 / (2pi) / 1e12   # √(eV/Å²/amu) → THz
const I3 = Matrix{Float64}(I, 3, 3)

"json 읽기와 쓰기 (JSON3.jl). readjson의 결과는 문자열이나 Symbol 키로 인덱싱한다"
readjson(path) = JSON3.read(read(path, String))
writejson(path, d) = open(io -> JSON3.pretty(io, JSON3.write(d)), path, "w")

"프로세스 CPU 시간(초). PythonCall은 파이썬을 같은 프로세스에서 돌리므로 time.process_time()이 Julia와 파이썬 몫을 함께 센다.
파이썬 Lab이 잰 것과 같은 양이다."
cputime() = pyconvert(Float64, pyimport("time").process_time())

# ================================================================ 2. PythonCall 다리
# PythonCall이 쓸 파이썬은 환경 변수로 고른다(CondaPkg를 끄고 기존 가상환경을 쓴다).
#   JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/path/to/venv/bin/python julia lab_10_k.jl
# 파이썬 Lab과 같이 MACE-MP-0와 CHGNet은 한 환경(v10), SevenNet과 MatterSim은 다른 환경(v10b)에 있었다.
# 한 Julia 세션은 파이썬 인터프리터 하나만 쓰므로, Lab 10.5는 모델마다 맞는 환경으로 따로 실행한다.
const np = pyimport("numpy")

"torch가 있으면 스레드를 1개로 묶는다(공유 코어). torch가 없는 환경에서는 아무것도 하지 않는다."
function torch_one_thread()
    try
        pyimport("torch").set_num_threads(1)
    catch
    end
    return nothing
end

"numpy 배열(Py) → Julia 배열. 인덱스의 뜻은 numpy와 같다(a[i, j, k] = x[i-1, j-1, k-1])."
jarr(T, x) = pyconvert(Array{T}, np.asarray(x))

"Julia (Z, R 3 x n, H 3 x 3) → ase.Atoms. 전치에 주의한다."
function to_ase(Z, R, H; pbc = true)
    ase = pyimport("ase")
    if H === nothing
        return ase.Atoms(; numbers = np.array(collect(Z)), positions = np.array(permutedims(R)))
    end
    return ase.Atoms(; numbers = np.array(collect(Z)), positions = np.array(permutedims(R)),
                     cell = np.array(permutedims(H)), pbc = pbc)
end
"ase.Atoms → (Z, R 3 x n, H 3 x 3)"
from_ase(a) = (pyconvert(Vector{Int}, a.numbers), permutedims(jarr(Float64, a.positions)),
               permutedims(jarr(Float64, a.cell.array)))

"""ASE 계산기 만들기. name ∈ (:mace, :chgnet, :sevennet, :mattersim, :emt). (calc, net)을 돌려준다.
net은 매개변수를 세는 데 쓰는 torch 모듈(EMT는 nothing). 파이썬 labs/103_common.py의 make_calc와 같다."""
function make_calc(name::Symbol; dtype = "float64")
    torch_one_thread()
    if name == :mace
        calc = pyimport("mace.calculators").MACECalculator(; model_paths = joinpath(DATA, "mace_mp0_small.model"),
                                                            device = "cpu", default_dtype = dtype)
        return calc, calc.models[0]
    elseif name == :chgnet
        m = pyimport("chgnet.model")
        net = m.CHGNet.load(; verbose = false, use_device = "cpu")
        return m.CHGNetCalculator(; model = net, use_device = "cpu"), net
    elseif name == :sevennet
        calc = pyimport("sevenn.calculator").SevenNetCalculator("7net-0"; device = "cpu")
        return calc, calc.model
    elseif name == :mattersim
        try
            pyimport("loguru").logger.remove()
        catch
        end
        calc = pyimport("mattersim.forcefield").MatterSimCalculator(; device = "cpu")
        return calc, calc.potential.model
    elseif name == :emt
        return pyimport("ase.calculators.emt").EMT(), nothing
    end
    error("unknown calculator $name")
end
"MACE-MP-0 small 오라클(DFT의 대역). §10.2의 Lab은 float64, §10.4의 Lab은 float32를 썼다."
oracle(dtype = "float32") = first(make_calc(:mace; dtype))

"torch 모듈의 매개변수 수와 dtype 문자열"
function torch_param_info(net)
    n = sum(pyconvert(Int, p.numel()) for p in net.parameters())
    dt = replace(pyconvert(String, pystr(first(net.parameters()).dtype)), "torch." => "")
    return n, dt
end

"""ASE 계산기로 에너지(eV), 힘(3 x n, eV/Å), 응력(3 x 3, eV/Å³, ASE 부호 규약: 인장이 양수)."""
function calc_efs(calc, Z, R, H; forces = true, stress = false, pbc = true)
    a = to_ase(Z, R, H; pbc); a.calc = calc
    E = pyconvert(Float64, a.get_potential_energy())
    F = forces ? permutedims(jarr(Float64, a.get_forces())) : nothing
    S = stress ? jarr(Float64, a.get_stress(; voigt = false)) : nothing
    return E, F, S
end
calc_energy(calc, Z, R, H; pbc = true) = first(calc_efs(calc, Z, R, H; forces = false, pbc))

"ase.build.bulk(...).repeat((n, n, n))를 Julia (Z, R, H)로"
function bulk_cell(sym, crystal, a; n = 1, cubic = true)
    at = pyimport("ase.build").bulk(sym, crystal; a = a, cubic = cubic).repeat((n, n, n))
    return from_ase(at)
end
"64원자 다이아몬드 Si (§10.4의 계). 오라클의 격자상수 5.464 Å"
const A0_SI = 5.464
si64(a = A0_SI) = bulk_cell("Si", "diamond", a; n = 2)

# ---------------------------------------------------------------- 자료: 프레임
"구조 하나. pos와 F는 3 x n, H는 3 x 3(열 = 격자 벡터). 라벨이 없으면 E = NaN, F = nothing."
Base.@kwdef mutable struct Frame
    pos::Matrix{Float64}
    H::Matrix{Float64}
    Z::Vector{Int}
    E::Float64 = NaN
    F::Union{Nothing,Matrix{Float64}} = nothing
    T::Float64 = 0.0
    step::Int = 0
    tag::String = ""
    dev::Float64 = NaN            # 위원회의 힘 불일치 ε_F (식 10.4.1)
    traj::Int = 0
    failed::Bool = false
end
natoms(f::Frame) = length(f.Z)
Base.copy(f::Frame) = Frame(; pos = copy(f.pos), H = copy(f.H), Z = copy(f.Z), E = f.E, F = f.F === nothing ? nothing : copy(f.F),
                             T = f.T, step = f.step, tag = f.tag, dev = f.dev, traj = f.traj, failed = f.failed)

"labs/104_common.py의 save_frames가 쓴 npz(savez_compressed)를 numpy로 읽는다. NPZ.jl 대신 numpy를 쓰는 것은
압축 파일과 문자열 배열(tag)을 확실히 읽기 위해서다."
function load_frames(name)
    d = np.load(joinpath(DATA, name); allow_pickle = false)
    pos = jarr(Float64, d["pos"]); cell = jarr(Float64, d["cell"]); Z = jarr(Int, d["Z"])
    E = jarr(Float64, d["E"]); F = jarr(Float64, d["F"]); T = jarr(Float64, d["T"]); st = jarr(Int, d["step"])
    tag = pyconvert(Vector{String}, d["tag"].tolist())
    return [Frame(; pos = permutedims(pos[k, :, :]), H = permutedims(cell[k, :, :]), Z = Z[k, :], E = E[k],
                  F = permutedims(F[k, :, :]), T = T[k], step = st[k], tag = tag[k]) for k in eachindex(E)]
end

"§10.2의 공용 자료 data/102_si8.npz(8원자 Si 셀 600개, 앞 400개가 학습). R, F: 3 x 8 x n, H, S: 3 x 3 x n"
function load_si8()
    d = np.load(joinpath(DATA, "102_si8.npz"))
    n = pyconvert(Int, d["n_train"])
    R = permutedims(jarr(Float64, d["R"]), (3, 2, 1)); H = permutedims(jarr(Float64, d["cell"]), (3, 2, 1))
    F = permutedims(jarr(Float64, d["F"]), (3, 2, 1)); S = permutedims(jarr(Float64, d["S"]), (2, 3, 1))
    E = jarr(Float64, d["E"])
    pick(r) = (R = R[:, :, r], H = H[:, :, r], E = E[r], F = F[:, :, r], S = S[:, :, r])
    return pick(1:n), pick(n+1:length(E))
end

"라벨(오라클 호출). 파이썬 104_common.label과 같다: 프레임마다 에너지와 힘"
function label(calc, frames)
    return map(frames) do f
        E, F, _ = calc_efs(calc, f.Z, f.pos, f.H)
        g = copy(f); g.E = E; g.F = F; g
    end
end

# ================================================================ 3. 주기 이웃 목록
"""주기 이웃 목록 (i, j, S): |r_j - r_i + H S| < rc인 모든 순서쌍(같은 원자의 다른 이미지도 포함). ASE의 neighbor_list('ijS')와
같은 쌍의 집합을 낸다(순서는 다르다). 좌표를 셀 안으로 접어 이미지 범위를 정하고, 접기 전 좌표에 맞게 S를 되돌린다."""
function neighbor_list(R::AbstractMatrix, H::AbstractMatrix, rc)
    n = size(R, 2); Hi = inv(H)
    f = Hi * R; nimg = floor.(f); Rw = H * (f .- nimg)
    nk = [ceil(Int, rc * norm(Hi[k, :])) for k in 1:3]           # 면 사이 거리 d_k = 1/|H⁻¹의 k행|
    I_, J_ = Int[], Int[]; S_ = Vector{Vector{Float64}}()
    rc2 = rc^2
    for a in -nk[1]:nk[1], b in -nk[2]:nk[2], c in -nk[3]:nk[3]
        sw = [a, b, c]; sh = H * sw
        for j in 1:n, i in 1:n
            (a == 0 && b == 0 && c == 0 && i == j) && continue
            d1 = Rw[1, j] - Rw[1, i] + sh[1]; d2 = Rw[2, j] - Rw[2, i] + sh[2]; d3 = Rw[3, j] - Rw[3, i] + sh[3]
            if d1^2 + d2^2 + d3^2 < rc2
                push!(I_, i); push!(J_, j); push!(S_, sw .- nimg[:, j] .+ nimg[:, i])
            end
        end
    end
    return I_, J_, isempty(S_) ? zeros(3, 0) : reduce(hcat, S_)
end
"쌍 거리만 (최소 거리, 배위수 등)"
function pair_distances(R, H, rc)
    i, j, S = neighbor_list(R, H, rc)
    return [norm(R[:, j[e]] - R[:, i[e]] + H * S[:, e]) for e in eachindex(i)]
end
min_dist(R, H; rc = 2.6) = (d = pair_distances(R, H, rc); isempty(d) ? 9.9 : minimum(d))

"""배치 그래프. 셀 b의 원자는 arange[b], 간선은 erange[b]에 모여 있다. i, j는 배치 전체의 원자 번호.
stress = true면 모델 입력 x의 끝에 셀마다 3 x 3 변형 ε를 붙인다(Derivation Box 10.2)."""
struct BatchGraph
    i::Vector{Int}
    j::Vector{Int}
    S::Matrix{Float64}
    H::Array{Float64,3}
    Z::Vector{Int}
    batch::Vector{Int}
    arange::Vector{UnitRange{Int}}
    erange::Vector{UnitRange{Int}}
    nb::Int
    N::Int
    stress::Bool
end
"셀들의 (Z, R, H)로 배치 그래프. lists에 셀마다 미리 만든 이웃 목록을 넘기면 다시 계산하지 않는다(학습에서 쓴다)."
function batch_graph(Zs, Rs, Hs, rc; stress = false, lists = nothing)
    I_, J_, S_, ar, er, bt = Int[], Int[], Matrix{Float64}[], UnitRange{Int}[], UnitRange{Int}[], Int[]
    oa = 0; oe = 0
    for b in eachindex(Rs)
        i, j, S = lists === nothing ? neighbor_list(Rs[b], Hs[b], rc) : lists[b]
        n = size(Rs[b], 2)
        append!(I_, i .+ oa); append!(J_, j .+ oa); push!(S_, S)
        push!(ar, oa+1:oa+n); push!(er, oe+1:oe+length(i)); append!(bt, fill(b, n))
        oa += n; oe += length(i)
    end
    H = cat(Hs...; dims = 3)
    return BatchGraph(I_, J_, reduce(hcat, S_), H, reduce(vcat, Zs), bt, ar, er, length(Rs), oa, stress)
end
batch_graph(frames::AbstractVector{Frame}, rc; kw...) =
    batch_graph([f.Z for f in frames], [f.pos for f in frames], [f.H for f in frames], rc; kw...)

"모델 입력 x에서 (변형을 가한) 좌표 3 x N과 셀 3 x 3 x nb. stress가 아니면 x = vec(R) 그대로다."
function strained_geometry(g::BatchGraph, x)
    R = reshape(x[1:3*g.N], 3, g.N)
    g.stress || return R, [g.H[:, :, b] for b in 1:g.nb]
    eps = reshape(x[3*g.N+1:end], 3, 3, g.nb)
    sym = [(eps[:, :, b] .+ transpose(eps[:, :, b])) ./ 2 for b in 1:g.nb]          # 대칭 변형
    Rd = reduce(hcat, [(I3 .+ sym[b]) * R[:, g.arange[b]] for b in 1:g.nb])        # r → (I + ε) r
    Hd = [(I3 .+ sym[b]) * g.H[:, :, b] for b in 1:g.nb]                             # H → (I + ε) H
    return Rd, Hd
end
"간선 벡터 r_ij = R_j - R_i + H S_ij (3 x E). NNlib.gather의 역전파는 scatter다."
function edge_vectors(g::BatchGraph, R, Hd)
    sh = reduce(hcat, [Hd[b] * g.S[:, g.erange[b]] for b in 1:g.nb])
    return NNlib.gather(R, g.j) .- NNlib.gather(R, g.i) .+ sh
end
"간선 → 원자 합(index_add): F x E → F x N (마지막 축 기준)"
scatter_to(x, idx, n) = NNlib.scatter(+, x, idx; dstsize = (size(x)[1:end-1]..., n))

# ================================================================ 4. 학생 모델 (Lux.jl)
cosine_cutoff(r, rc) = 0.5 .* (cos.(pi .* r ./ rc) .+ 1) .* (r .< rc)
"Bessel 기저 √(2/rc) sin(kπr/rc)/r, k = 1..n (n x E)"
bessel_rbf(r, n, rc) = sqrt(2 / rc) .* sin.((1:n) .* (pi / rc) .* reshape(r, 1, :)) ./ reshape(r, 1, :)
"Gaussian 기저 exp(-γ (r - μ_k)²) (n x E)"
gaussian_rbf(r, mu, gamma) = exp.(-gamma .* (reshape(r, 1, :) .- mu) .^ 2)
"shifted softplus ln(0.5 eˣ + 0.5), SchNet의 활성화"
ssp(x) = softplus(x) - oftype(x, log(2))

"PyTorch nn.Linear의 기본 초기화(가중치, 편향 모두 U(±1/√fan_in)), Float64"
tu(nin) = (rng, dims...) -> (rand(rng, Float64, dims...) .* 2 .- 1) ./ sqrt(nin)
ldense(nin, nout, act = identity; bias = true) = Dense(nin => nout, act; init_weight = tu(nin), init_bias = tu(nin), use_bias = bias)
"원자마다 같은 학습 가능 벡터 h0 (torch.randn(F) * 0.5): 1 → F, 편향 없는 Dense를 ones(1, N)에 쓴다"
h0_layer(F) = Dense(1 => F; use_bias = false, init_weight = (rng, d...) -> randn(rng, Float64, d...) .* 0.5)

# ---------------------------------------------------------------- InvNet (SchNet형, 불변)
struct InvBlock{A,B,C} <: Lux.AbstractLuxContainerLayer{(:filt, :lin, :upd)}
    filt::A
    lin::B
    upd::C
end
InvBlock(F, nrbf) = InvBlock(Chain(ldense(nrbf, F, swish), ldense(F, F)), ldense(F, F), Chain(ldense(F, F, swish), ldense(F, F)))
function (blk::InvBlock)(x, ps, st)
    h, w, g = x
    W, _ = blk.filt(w, ps.filt, st.filt)
    a, _ = blk.lin(h, ps.lin, st.lin)
    m = scatter_to(W .* NNlib.gather(a, g.j), g.i, g.N)            # m_i = Σ_j filt(w_ij) ⊙ lin(h)_j
    u, _ = blk.upd(m, ps.upd, st.upd)
    return (h .+ u, w, g), st
end

# ---------------------------------------------------------------- EqNet (PaiNN형, 스칼라 s: F x N, 벡터 v: F x 3 x N)
struct EqBlock{A,B,C,D,E} <: Lux.AbstractLuxContainerLayer{(:filt, :phi, :U, :V, :upd)}
    filt::A
    phi::B
    U::C
    V::D
    upd::E
end
EqBlock(F, nrbf) = EqBlock(ldense(nrbf, 3F), Chain(ldense(F, F, swish), ldense(F, 3F)), ldense(F, F; bias = false),
                           ldense(F, F; bias = false), Chain(ldense(2F, F, swish), ldense(F, 3F)))
"F x 3 x N 벡터 특징에 채널 방향 선형층(편향 없음)을 쓴다"
function vlinear(layer, ps, st, v)
    F, _, N = size(v)
    y, _ = layer(reshape(v, F, 3N), ps, st)
    return reshape(y, size(y, 1), 3, N)
end
function (blk::EqBlock)(x, ps, st)
    s, v, w, rhat, g = x
    F = size(s, 1); N = g.N; E = length(g.i)
    p, _ = blk.phi(s, ps.phi, st.phi); fw, _ = blk.filt(w, ps.filt, st.filt)
    y = NNlib.gather(p, g.j) .* fw                                      # 3F x E
    ds, dvv, dvs = y[1:F, :], y[F+1:2F, :], y[2F+1:3F, :]
    s = s .+ scatter_to(ds, g.i, N)
    dv = NNlib.gather(v, g.j) .* reshape(dvv, F, 1, E) .+ reshape(rhat, 1, 3, E) .* reshape(dvs, F, 1, E)
    v = v .+ scatter_to(dv, g.i, N)
    Uv = vlinear(blk.U, ps.U, st.U, v); Vv = vlinear(blk.V, ps.V, st.V, v)
    a, _ = blk.upd(vcat(s, reshape(sum(abs2, Vv; dims = 2), F, N)), ps.upd, st.upd)   # |V v_i|²에 결합각이 들어 있다 (식 10.2.4)
    avv, asv, ass = a[1:F, :], a[F+1:2F, :], a[2F+1:3F, :]
    v = v .+ reshape(avv, F, 1, N) .* Uv
    s = s .+ asv .* reshape(sum(Uv .* Vv; dims = 2), F, N) .+ ass
    return (s, v, w, rhat, g), st
end

"""§10.2의 학생. kind = :inv(InvNet) 또는 :eq(EqNet). head = :energy(보존력) 또는 :direct(별도의 벡터 머리).
E = Σ_i (ε_i ⋅ escale + e0). graph는 현재 배치의 BatchGraph를 담는 상자다(미분 밖에서 바꾼다)."""
struct Student{H0,BK,O,FV} <: Lux.AbstractLuxContainerLayer{(:h0, :blocks, :out, :fvec)}
    h0::H0
    blocks::BK
    out::O
    fvec::FV
    kind::Symbol
    rc::Float64
    nrbf::Int
    head::Symbol
    e0::Base.RefValue{Float64}
    escale::Base.RefValue{Float64}
    graph::Base.RefValue{Any}
end
function Student(kind::Symbol, rc; F = 32, nrbf = 8, T = 2, head = :energy, e0 = 0.0)
    blocks = kind == :inv ? Chain([InvBlock(F, nrbf) for _ in 1:T]...) : Chain([EqBlock(F, nrbf) for _ in 1:T]...)
    return Student(h0_layer(F), blocks, Chain(ldense(F, F, swish), ldense(F, 1)), ldense(F, 1; bias = false),
                   kind, Float64(rc), nrbf, head, Ref(Float64(e0)), Ref(1.0), Ref{Any}(nothing))
end
modelkind(m::Student) = m.kind

"원자 특징 (s, v). InvNet은 v = nothing"
function features(m::Student, ps, st, x)
    g = m.graph[]
    R, Hd = strained_geometry(g, x)
    rij = edge_vectors(g, R, Hd)
    r = sqrt.(sum(abs2, rij; dims = 1))                                  # 1 x E
    w = bessel_rbf(vec(r), m.nrbf, m.rc) .* cosine_cutoff(r, m.rc)
    s0, _ = m.h0(ones(eltype(x), 1, g.N), ps.h0, st.h0)
    if modelkind(m) == :inv
        (s, _, _), _ = m.blocks((s0, w, g), ps.blocks, st.blocks)
        return s, nothing
    end
    F = size(s0, 1)
    v0 = zeros(eltype(x), F, 3, g.N)
    (s, v, _, _, _), _ = m.blocks((s0, v0, w, rij ./ r, g), ps.blocks, st.blocks)
    return s, v
end
"Lux 호출: 입력 x(평평한 좌표, stress면 끝에 ε) → 셀마다 에너지 (nb)"
function (m::Student)(x, ps, st)
    g = m.graph[]
    s, _ = features(m, ps, st, x)
    e, _ = m.out(s, ps.out, st.out)
    eps_i = vec(e) .* m.escale[] .+ m.e0[]
    return NNlib.scatter(+, eps_i, g.batch; dstsize = (g.nb,)), st
end
"직접 힘 머리: F_i = W v_i (3 x N). 어떤 에너지의 기울기도 아니다."
function direct_forces(m::Student, ps, st, x)
    s, v = features(m, ps, st, x)
    F = size(v, 1)
    y, _ = m.fvec(reshape(v, F, :), ps.fvec, st.fvec)
    return reshape(y, 3, :)
end

# ---------------------------------------------------------------- SchNetS (Ch.7 labs/72b_schnet.py의 주기판)
struct SchBlock{A,B,C} <: Lux.AbstractLuxContainerLayer{(:filt, :in2f, :out)}
    filt::A
    in2f::B
    out::C
end
SchBlock(F, nrbf) = SchBlock(Chain(ldense(nrbf, F, ssp), ldense(F, F)), ldense(F, F; bias = false), Chain(ldense(F, F, ssp), ldense(F, F)))
function (blk::SchBlock)(x, ps, st)
    h, rbf, fc, g = x
    W, _ = blk.filt(rbf, ps.filt, st.filt)
    a, _ = blk.in2f(h, ps.in2f, st.in2f)
    m = scatter_to(NNlib.gather(a, g.j) .* (W .* fc), g.i, g.N)
    o, _ = blk.out(m, ps.out, st.out)
    return (h .+ o, rbf, fc, g), st
end
"§10.4의 학생 SchNet(특징 32, Gaussian 기저 16, 상호작용 2, rc = 4 Å; 매개변수 13,217개). e_shift[Z]는 원소별 기준 에너지 E0"
struct SchNetS{EM,BK,HD} <: Lux.AbstractLuxContainerLayer{(:emb, :blocks, :head)}
    emb::EM
    blocks::BK
    head::HD
    mu::Vector{Float64}
    gamma::Float64
    rc::Float64
    e_shift::Vector{Float64}
    graph::Base.RefValue{Any}
end
function SchNetS(; F = 32, nrbf = 16, rc = 4.0, T = 2, maxz = 100)
    mu = Float64.(collect(range(0.0f0, Float32(rc); length = nrbf)))   # torch.linspace(float32)와 같은 값
    emb = Embedding(maxz => F; init_weight = (rng, d...) -> randn(rng, Float64, d...))
    return SchNetS(emb, Chain([SchBlock(F, nrbf) for _ in 1:T]...), Chain(ldense(F, F ÷ 2, ssp), ldense(F ÷ 2, 1)),
                   mu, 0.5 / (mu[2] - mu[1])^2, Float64(rc), zeros(maxz), Ref{Any}(nothing))
end
function (m::SchNetS)(x, ps, st)
    g = m.graph[]
    R, Hd = strained_geometry(g, x)
    rij = edge_vectors(g, R, Hd)
    r = vec(sqrt.(sum(abs2, rij; dims = 1) .+ 1e-12))
    h, _ = m.emb(g.Z .+ 1, ps.emb, st.emb)                                # 원자번호 Z는 0부터 센 행(파이썬과 같다)
    (h, _, _, _), _ = m.blocks((h, gaussian_rbf(r, m.mu, m.gamma), reshape(cosine_cutoff(r, m.rc), 1, :), g), ps.blocks, st.blocks)
    e, _ = m.head(h, ps.head, st.head)
    return NNlib.scatter(+, vec(e) .+ m.e_shift[g.Z], g.batch; dstsize = (g.nb,)), st
end
cutoff(m::Student) = m.rc
cutoff(m::SchNetS) = m.rc

"매개변수 수 (fvec는 보존 모델에서도 세므로 EqNet은 25,921개, InvNet은 10,145개, SchNetS는 13,217개)"
nparams(m) = Lux.parameterlength(m)

# ---------------------------------------------------------------- 에너지, 힘, 응력
"""(E (nb), F (3 x N), S (3 x 3 x nb) 또는 nothing). 보존 모델은 F = -∂E/∂R, 응력은 σ = (1/V) ∂E/∂ε (ASE 부호).
학습 손실 안(바깥 Zygote.gradient 안)에서 불리면, Lux가 안쪽 Zygote.gradient를 ForwardDiff-over-Zygote로 바꾸어 준다
(Lux의 중첩 AD, Ch.7 mlip_common.jl의 energy_forces와 같은 방식). g는 미분 밖에서 m.graph[]에 넣어 두어야 한다."""
function efs(m, ps, st, R::AbstractMatrix)
    g = m.graph[]
    x = g.stress ? vcat(vec(R), zeros(eltype(R), 9 * g.nb)) : vec(R)
    sm = StatefulLuxLayer{true}(m, ps, st)
    E = sm(x)
    if m isa Student && m.head == :direct
        return E, direct_forces(m, ps, st, x), nothing
    end
    gx = only(Zygote.gradient(sum ∘ sm, x))
    F = -reshape(gx[1:3*g.N], 3, g.N)
    g.stress || return E, F, nothing
    V = [abs(det(g.H[:, :, b])) for b in 1:g.nb]
    return E, F, reshape(gx[3*g.N+1:end], 3, 3, g.nb) ./ reshape(V, 1, 1, :)
end
"그래프를 만들어 붙이고 efs를 부른다(미분 밖에서 쓰는 평가용)"
function predict(m, ps, st, Zs, Rs, Hs; stress = false, lists = nothing)
    m.graph[] = batch_graph(Zs, Rs, Hs, cutoff(m); stress, lists)
    return efs(m, ps, st, reduce(hcat, Rs))
end
predict(m, ps, st, frames::AbstractVector{Frame}; kw...) =
    predict(m, ps, st, [f.Z for f in frames], [f.pos for f in frames], [f.H for f in frames]; kw...)
"모델 하나로 구조 하나의 (E, F)"
student_ef(m, ps, st, Z, R, H) = (r = predict(m, ps, st, [Z], [R], [H]); (r[1][1], r[2]))

"Hessian ∂²E/∂R∂R (3n x 3n) by 이중 자동미분(Zygote.hessian = ForwardDiff over Zygote)"
function hessian_ad(m, ps, st, Z, R, H)
    m.graph[] = batch_graph([Z], [R], [H], cutoff(m))
    sm = StatefulLuxLayer{true}(m, ps, st)
    return Zygote.hessian(x -> sum(sm(x)), vec(R))
end

# ---------------------------------------------------------------- 학생 만들기, 저장, 읽기
"모델, 매개변수, 상태 (Float64). seed는 Julia Xoshiro 시드이므로 초기값은 파이썬과 다르다."
function setup_model(m; seed = 0)
    ps, st = Lux.setup(Xoshiro(seed), m)
    return m, ps, st
end
"§10.4의 학생: SchNetS, 원자당 기준 에너지 e0 (labs/104_common.new_student)"
function new_student(seed, e0)
    m = SchNetS(); m.e_shift[14] = e0
    return setup_model(m; seed)
end
save_model(path, m, ps, st) = serialize(path, (m, ps, st))
load_model(path) = deserialize(path)

# ================================================================ 5. 학습과 평가
"torch CosineAnnealingLR(T_max = total, eta_min)의 s번째(0부터) 학습률"
cosine_lr(lr0, s, total; eta_min = 0.0) = eta_min + (lr0 - eta_min) * (1 + cos(pi * s / total)) / 2

"""§10.2의 학습 (labs/102_train.py). data = load_si8()의 묶음, idx = 학습 셀 번호. 매 단계 배치 bs개를 비복원으로 뽑는다.
L = wE ⟨((E - E_ref)/n)²⟩ + wF ⟨(F - F_ref)²⟩ + wS ⟨(σ - σ_ref)²⟩. Adam, 코사인 감쇠. CPU 시간(초)을 돌려준다."""
function train_si8!(m, ps, st, data, idx; steps = 1500, bs = 16, lr = 3e-3, wE = 1.0, wF = 1.0, wS = 0.0, seed = 0, noiseF = 0.0)
    rng = Xoshiro(seed)
    nat = size(data.R, 2)
    Fref = noiseF > 0 ? data.F[:, :, idx] .+ noiseF .* randn(rng, size(data.F[:, :, idx])) : data.F[:, :, idx]
    lists = [neighbor_list(data.R[:, :, k], data.H[:, :, k], m.rc) for k in idx]   # 셀마다 한 번 (파이썬 Graph와 같다)
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    t0 = cputime(); n = length(idx)
    for it in 0:steps-1
        b = randperm(rng, n)[1:min(bs, n)]
        Rs = [data.R[:, :, idx[k]] for k in b]
        m.graph[] = batch_graph(fill(fill(14, nat), length(b)), Rs, [data.H[:, :, idx[k]] for k in b], m.rc;
                                stress = wS > 0, lists = lists[b])
        Rb = reduce(hcat, Rs); Eb = data.E[idx[b]]; Fb = reshape(Fref[:, :, b], 3, :); Sb = data.S[:, :, idx[b]]
        gs = only(Zygote.gradient(ps) do p
            if wF == 0 && wS == 0 && m.head == :energy
                E = StatefulLuxLayer{true}(m, p, st)(vec(Rb))
                return wE * mean(abs2, (E .- Eb) ./ nat)
            end
            E, F, S = efs(m, p, st, Rb)
            l = wE * mean(abs2, (E .- Eb) ./ nat)
            wF > 0 && (l += wF * mean(abs2, F .- Fb))
            wS > 0 && (l += wS * mean(abs2, S .- Sb))
            return l
        end)
        Optimisers.adjust!(opt, cosine_lr(lr, it, steps))
        opt, ps = Optimisers.update!(opt, ps, gs)
    end
    return ps, cputime() - t0
end

"§10.2의 평가: 에너지 MAE(meV/atom), 힘 MAE와 RMSE(meV/Å), 그리고 예측값. chunk개씩 나누어 계산한다."
function evaluate_si8(m, ps, st, data; chunk = 50, stress = false)
    n = length(data.E); nat = size(data.R, 2)
    Ep, Fp, Sp = Float64[], Matrix{Float64}[], Array{Float64,3}[]
    for a in 1:chunk:n
        r = a:min(n, a + chunk - 1)
        E, F, S = predict(m, ps, st, fill(fill(14, nat), length(r)), [data.R[:, :, k] for k in r], [data.H[:, :, k] for k in r]; stress)
        append!(Ep, E); push!(Fp, F); stress && push!(Sp, S)
    end
    F = reshape(reduce(hcat, Fp), 3, nat, n)
    out = (E_mae = 1e3 * mean(abs.((Ep .- data.E) ./ nat)), F_mae = 1e3 * mean(abs.(F .- data.F)),
           F_rmse = 1e3 * sqrt(mean(abs2, F .- data.F)), E = Ep, F = F)
    stress || return out
    return merge(out, (S_mae = GPA * mean(abs.(cat(Sp...; dims = 3) .- data.S)),))
end

"""§10.4의 학습 (labs/104_common.train): 에포크마다 순열, 배치 bs, L = wE ⟨((E - E_ref)/n)²⟩ + wF ⟨(F - F_ref)²⟩,
Adam, 코사인 감쇠(eta_min = 0.05 lr, 전체 갱신 수에 걸쳐). 이전 매개변수에서 이어서 학습할 수 있다(warm start)."""
function train_frames!(m, ps, st, frames, epochs; lr = 2e-3, wE = 1.0, wF = 10.0, bs = 8, seed = 0)
    rng = Xoshiro(seed)
    lists = [neighbor_list(f.pos, f.H, m.rc) for f in frames]
    nb = max(1, ceil(Int, length(frames) / bs)); total = epochs * nb
    opt = Optimisers.setup(Optimisers.Adam(lr), ps); s = 0
    for ep in 1:epochs
        idx = randperm(rng, length(frames))
        for k in 1:nb
            sel = idx[(k-1)*bs+1:min(k * bs, length(idx))]
            fr = frames[sel]
            m.graph[] = batch_graph(fr, m.rc; lists = lists[sel])
            Rb = reduce(hcat, [f.pos for f in fr]); Eb = [f.E for f in fr]; nat = [natoms(f) for f in fr]
            Fb = reduce(hcat, [f.F for f in fr])
            gs = only(Zygote.gradient(ps) do p
                E, F, _ = efs(m, p, st, Rb)
                wE * mean(abs2, (E .- Eb) ./ nat) + wF * mean(abs2, F .- Fb)
            end)
            Optimisers.adjust!(opt, cosine_lr(lr, s, total; eta_min = 0.05lr)); s += 1
            opt, ps = Optimisers.update!(opt, ps, gs)
        end
    end
    return ps
end

"RMSE: 원자당 에너지(meV/atom), 힘 성분(meV/Å) (labs/104_common.errors)"
function errors(m, ps, st, frames)
    de, df = Float64[], Float64[]
    for a in 1:16:length(frames)
        fr = frames[a:min(end, a + 15)]
        E, F, _ = predict(m, ps, st, fr)
        append!(de, (E .- [f.E for f in fr]) ./ [natoms(f) for f in fr])
        append!(df, vec(F .- reduce(hcat, [f.F for f in fr])))
    end
    return 1e3 * sqrt(mean(abs2, de)), 1e3 * sqrt(mean(abs2, df))
end

# ================================================================ 6. MD
"Maxwell–Boltzmann 속도(Å/fs), 총 운동량 0. exact = true면 자유도 3n - 3로 온도를 정확히 맞춘다(labs/102_nve.py)."
function mb_velocities(rng, n, T, mass; exact = false)
    v = randn(rng, 3, n) .* sqrt(KB * T / mass)
    v .-= mean(v; dims = 2)
    exact && (v .*= sqrt(KB * T * (3n - 3) / (mass * sum(abs2, v))))
    return v
end
kinetic(v, mass) = 0.5 * mass * sum(abs2, v)
temperature(v, mass) = 2 * kinetic(v, mass) / ((3 * size(v, 2) - 3) * KB)

"""속도 Verlet NVE. force(R) → (E, F). mass는 eV fs²/Å² 단위. record(s, R, v, E)는 매 걸음 불린다.
E가 유한하지 않으면 멈추고 마지막 걸음 번호를 돌려준다."""
function velocity_verlet(force, R0, v0, mass, dt, nsteps; record = (s, R, v, E) -> nothing)
    R = copy(R0); v = copy(v0)
    E, F = force(R)
    record(0, R, v, E)
    for s in 1:nsteps
        v .+= 0.5 * dt .* F ./ mass
        R .+= dt .* v
        E, F = force(R)
        v .+= 0.5 * dt .* F ./ mass
        isfinite(E) || return R, v, s
        record(s, R, v, E)
    end
    return R, v, nsteps
end

"위원회의 힘 불일치 ε_F = max_i √(⟨‖F_i^(k) - F̄_i‖²⟩_k) (식 10.4.1, DP-GEN). Fs: 구성원별 3 x n 힘의 벡터"
function force_deviation(Fs)
    Fm = mean(Fs)
    return maximum(sqrt.(vec(mean([sum(abs2, F .- Fm; dims = 1) for F in Fs]))))
end

"""BAOAB Langevin MD (labs/104_common.langevin). models = [(m, ps, st), ...]의 첫 구성원이 MD를 몬다.
committee = true면 매 기록마다 모든 구성원의 힘으로 ε_F를 잰다. 실패 = Si–Si < 1.9 Å 또는 |F| > 25 eV/Å.
(frames, 실패한 걸음 또는 nothing)을 돌려준다."""
function langevin(models, pos0, H, Z, T, nsteps; seed = 0, dt = 2.0, gamma = 0.02, every = 1, committee = false,
                  keep_fail = false, fail_dmin = 1.9)
    mass = MSI * AMU; rng = Xoshiro(seed); n = length(Z)
    m, ps, st = models[1]
    v = randn(rng, 3, n) .* sqrt(KB * T / mass); v .-= mean(v; dims = 2)
    x = copy(pos0)
    F = student_ef(m, ps, st, Z, x, H)[2]
    c1 = exp(-gamma * dt); c2 = sqrt((1 - c1^2) * KB * T / mass)
    frames = Frame[]; fail = nothing
    devof(x) = force_deviation([student_ef(mm, pp, ss, Z, x, H)[2] for (mm, pp, ss) in models])
    for s in 1:nsteps
        v .+= 0.5 * dt .* F ./ mass                 # B
        x .+= 0.5 * dt .* v                         # A
        v .= c1 .* v .+ c2 .* randn(rng, 3, n)      # O
        x .+= 0.5 * dt .* v                         # A
        F = student_ef(m, ps, st, Z, x, H)[2]
        v .+= 0.5 * dt .* F ./ mass                 # B
        if !all(isfinite, x) || min_dist(x, H) < fail_dmin || maximum(abs, F) > 25.0
            fail = s
            if keep_fail && all(isfinite, x)        # 학생이 무너진 구조도 후보에 넣는다
                fr = Frame(; pos = copy(x), H, Z, step = s, failed = true)
                committee && (fr.dev = devof(x))
                push!(frames, fr)
            end
            break
        end
        if s % every == 0
            fr = Frame(; pos = copy(x), H, Z, step = s)
            committee && (fr.dev = devof(x))
            push!(frames, fr)
        end
    end
    return frames, fail
end

# ================================================================ 7. 결정론적 도구
"1차 최소제곱 기울기 (numpy polyfit(x, y, 1)[0])"
function slope(x, y)
    xm = mean(x); ym = mean(y)
    return sum((x .- xm) .* (y .- ym)) / sum(abs2, x .- xm)
end
"선형 맞춤을 뺀 나머지의 표준편차(모집단 표준편차, numpy std와 같다)"
function detrended_std(x, y)
    b = slope(x, y); a = mean(y) - b * mean(x)
    return std(y .- (a .+ b .* x); corrected = false)
end
"선형 보간 (numpy interp, x는 증가 순서)"
function interp(x0, xs, ys)
    x0 <= xs[1] && return ys[1]; x0 >= xs[end] && return ys[end]
    k = searchsortedlast(xs, x0)
    return ys[k] + (ys[k+1] - ys[k]) * (x0 - xs[k]) / (xs[k+1] - xs[k])
end

"""3차 Birch–Murnaghan 맞춤. BM3의 E(V)는 x = V^(-2/3)의 3차식이므로, 계수를 선형 최소제곱으로 구하면 ASE의
EquationOfState(eos = 'birchmurnaghan')가 비선형으로 찾는 것과 같은 최적점이다. (V0, E0, B0 (eV/Å³), B0′)을 돌려준다."""
function birch_murnaghan(V, E)
    x = V .^ (-2 / 3)
    c = hcat(ones(length(x)), x, x .^ 2, x .^ 3) \ E                 # E = c1 + c2 x + c3 x² + c4 x³
    d1(x) = c[2] + 2c[3] * x + 3c[4] * x^2
    d2(x) = 2c[3] + 6c[4] * x
    disc = sqrt(c[3]^2 - 3 * c[2] * c[4])                            # E′(x) = 0의 두 근 중 E″ > 0인 것
    xs = [(-c[3] + disc) / (3c[4]), (-c[3] - disc) / (3c[4])]
    x0 = xs[argmax([d2(z) for z in xs])]
    V0 = x0^(-3 / 2)
    E0 = c[1] + c[2] * x0 + c[3] * x0^2 + c[4] * x0^3
    B0 = 4 / 9 * d2(x0) * V0^(-7 / 3)                                # B = V d²E/dV², dx/dV = -(2/3) V^(-5/3)
    # B0′ = dB/dP = -(V/B) dB/dV. B(V) = V d²E/dV² = V [E″(x)(4/9)V^(-10/3) + E′(x)(10/9)V^(-8/3)]를 수치 미분한다
    Bof(V) = V * ((4 / 9) * d2(V^(-2 / 3)) * V^(-10 / 3) + d1(V^(-2 / 3)) * (10 / 9) * V^(-8 / 3))
    h = 1e-4 * V0
    Bp = -V0 * (Bof(V0 + h) - Bof(V0 - h)) / (2h) / B0
    return V0, E0, B0, Bp
end

"""유한차분 응력 σ_ab ≈ [E(I + ε⁺) - E(I - ε⁺)]/(2hV), ε⁺는 (a, b)와 (b, a)에 h/2씩 넣은 대칭 변형 (labs/102_stress.fd_stress).
energy(R, H)는 좌표 3 x n과 셀 3 x 3을 받는다."""
function fd_stress(energy, R, H; h = 1e-4)
    V = abs(det(H)); S = zeros(3, 3)
    for a in 1:3, b in 1:3
        e = zeros(3, 3); e[a, b] += h / 2; e[b, a] += h / 2
        Ep = energy((I3 + e) * R, (I3 + e) * H)
        Em = energy((I3 - e) * R, (I3 - e) * H)
        S[a, b] = (Ep - Em) / (2h) / V
    end
    return S
end

"""힘의 중심 유한차분 Hessian H_kl = -∂F_k/∂x_l (h = 10⁻³ Å), 그리고 대칭화 (H + Hᵀ)/2 (labs/102_nve.hessian_fd).
force(R)는 3 x n 힘을 돌려준다. 열의 순서는 vec(R)(원자마다 x, y, z)로, 파이썬의 R.reshape(-1)과 같다."""
function hessian_fd(force, R; h = 1e-3)
    n = length(R); Hm = zeros(n, n)
    for k in 1:n
        d = zeros(n); d[k] = h
        Fp = vec(force(reshape(vec(R) .+ d, 3, :))); Fm = vec(force(reshape(vec(R) .- d, 3, :)))
        Hm[:, k] = -(Fp .- Fm) ./ (2h)
    end
    return (Hm .+ Hm') ./ 2
end
"질량 가중 Hessian의 고유 진동수(THz). 음의 고유값은 음의 진동수로 표시한다(labs/102_nve.freqs_thz)"
function freqs_thz(Hm; mass = MSI)
    w2 = eigvals(Symmetric(Hm ./ mass))
    return sign.(w2) .* sqrt.(abs.(w2)) .* THZ
end

"""최원점 선택(FPS): 고른 집합까지의 최소 거리가 가장 먼 후보를 차례로 더한다(§10.4.1). Zs는 d x n(열 = 후보).
start는 1부터 센 번호. 동점이면 번호가 작은 쪽(numpy argmax와 같다)."""
function fps(Zs, n, start)
    sel = [start]
    d = vec(sqrt.(sum(abs2, Zs .- Zs[:, start]; dims = 1)))
    for _ in 1:n-1
        k = argmax(d); push!(sel, k)
        d = min.(d, vec(sqrt.(sum(abs2, Zs .- Zs[:, k]; dims = 1))))
    end
    return sel
end
"고른 집합까지 각 질의 점의 최근접 거리"
nn_dist(Zsel, Zq) = [minimum(vec(sqrt.(sum(abs2, Zsel .- Zq[:, q]; dims = 1)))) for q in axes(Zq, 2)]

"힘의 시간 자기상관 c(L) = ⟨F(t)⋅F(t+L)⟩_t / ⟨F(t)⋅F(t)⟩_t, L = 0..maxlag (Estimate Box 10.6). Fs: 프레임별 3 x n 힘"
function force_acf(Fs, maxlag)
    c = [mean(sum(Fs[t] .* Fs[t+L]) for t in 1:length(Fs)-L) for L in 0:maxlag]
    return c ./ c[1]
end

"균일 무작위 회전 행렬 (Gaussian 행렬의 QR, 부호 보정, det = +1)"
function random_rotation(rng)
    Fq = qr(randn(rng, 3, 3)); M = Matrix(Fq.Q) * Diagonal(sign.(diag(Fq.R)))
    det(M) < 0 && (M[:, 1] .*= -1)
    return M
end
