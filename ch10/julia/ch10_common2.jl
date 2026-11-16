# ch10_common2.jl: Lab 10.8–10.13이 ch10_common.jl에 더해 쓰는 부분
# ch10_common.jl(PythonCall 다리, 학생 SchNet, 학습, MD, Birch–Murnaghan 등)을 먼저 include하고, 그 위에 다음을 더한다.
# (1) 네 uMLIP 이름표와 인터프리터 고르기. 파이썬 모델이 설치된 가상환경이 둘이다(mace, chgnet은 v10, sevennet, mattersim은 v10b).
#     PythonCall은 처음 불러올 때 인터프리터를 정하므로, 이 파일을 include하기 전에 ENV["CH10_MODEL"]을 정한다.
# (2) 원자 구조 Cell: Z (n), R (3 x n, 열 = 원자), h (3 x 3, 열 = 격자 벡터). AtomsBase 대신 평범한 배열이다.
#     ASE의 positions (n, 3)와 cell(행 = 격자 벡터)의 전치이고, ch10_common.jl의 (Z, R, H) 약속과 같다.
# (3) ASE에 맡기는 셀 이완(FrechetCellFilter + FixSymmetry), 호출 횟수를 세는 계산기 묶음.
# (4) 직접 옮긴 수치 도구: ASE 3.29와 같은 BFGS(원자 위치만)와 FIRE, ASE 기본 'sj' 상태방정식,
#     유한 변위 힘 상수와 공약 q점의 동역학 행렬(phonopy와 같은 병진, 순열 대칭화), 겹침수로 모드 붙이기.
const PY_EXE = Dict("mace" => "/tmp/claude-0/v10/bin/python", "chgnet" => "/tmp/claude-0/v10/bin/python",
                    "sevennet" => "/tmp/claude-0/v10b/bin/python", "mattersim" => "/tmp/claude-0/v10b/bin/python")
if !haskey(ENV, "JULIA_PYTHONCALL_EXE")
    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"                 # conda를 쓰지 않고 기존 가상환경의 파이썬을 쓴다
    ENV["JULIA_PYTHONCALL_EXE"] = PY_EXE[get(ENV, "CH10_MODEL", "mace")]
end
include("ch10_common.jl")
torch_one_thread()
pyimport("warnings").filterwarnings("ignore")

const LABEL = Dict("mace" => "MACE-MP-0 small", "chgnet" => "CHGNet 0.3.0", "sevennet" => "SevenNet-0", "mattersim" => "MatterSim 1M")
const MODELS = ["mace", "chgnet", "sevennet", "mattersim"]

# ---------------------------------------------------------------- 단위
const SQRT_EV_A2_AMU_CM = 521.47                       # √(eV/(Å² amu))를 cm⁻¹로 (2π 포함; ch10_common.jl의 THZ × 33.356)
const MASS = Dict(14 => 28.0855, 38 => 87.62, 22 => 47.867, 8 => 15.9994)   # phonopy의 원자량 (Si는 104_common.py의 값)

# ---------------------------------------------------------------- 구조 (평범한 배열)
struct Cell
    Z::Vector{Int}
    R::Matrix{Float64}      # 3 x n
    h::Matrix{Float64}      # 3 x 3, 열 = 격자 벡터
end
Base.copy(s::Cell) = Cell(copy(s.Z), copy(s.R), copy(s.h))
natoms(s::Cell) = length(s.Z)
volume(s::Cell) = abs(det(s.h))
frac(s::Cell) = s.h \ s.R
"cell(fractional F (3 x n), 원자번호, 격자 h)"
from_frac(Z, F, h) = Cell(collect(Z), h * F, Matrix{Float64}(h))

"ASE의 repeat와 같은 순서: 바깥 고리가 첫째 격자 벡터. 원자 번호 = 셀 번호 × n₀ + κ (셀 번호 = m₁ r₂ r₃ + m₂ r₃ + m₃)"
function repeat_cell(s::Cell, r::NTuple{3,Int})
    R = similar(s.R, 3, 0); Z = Int[]
    for m1 in 0:r[1]-1, m2 in 0:r[2]-1, m3 in 0:r[3]-1
        R = hcat(R, s.R .+ s.h * [m1, m2, m3]); append!(Z, s.Z)
    end
    return Cell(Z, R, s.h * Diagonal(collect(r)))
end
repeat_cell(s::Cell, k::Int) = repeat_cell(s, (k, k, k))

"입방정 SrTiO₃ 5원자 셀(Sr, Ti, O, O, O: ASE 스크립트와 같은 순서)"
sto(a; rep = 1) = (c = from_frac([38, 22, 8, 8, 8], [0 0.5 0.5 0.5 0; 0 0.5 0.5 0 0.5; 0 0.5 0 0.5 0.5], a * Matrix(I, 3, 3));
                   rep > 1 ? repeat_cell(c, rep) : c)
"다이아몬드 Si 8원자 입방 셀(ase.build.bulk('Si', 'diamond', cubic=True)와 같은 순서)"
function si_cubic(a; rep = 1)
    F = [0 0.25 0 0.25 0.5 0.75 0.5 0.75; 0 0.25 0.5 0.75 0 0.25 0.5 0.75; 0 0.25 0.5 0.75 0.5 0.75 0 0.25]
    c = from_frac(fill(14, 8), F, a * Matrix(I, 3, 3))
    return rep isa Int ? (rep > 1 ? repeat_cell(c, rep) : c) : repeat_cell(c, rep)
end
"원자 k를 지운 구조"
delete_atom(s::Cell, k) = (keep = [i for i in 1:natoms(s) if i != k]; Cell(s.Z[keep], s.R[:, keep], copy(s.h)))

"최소 영상 변위 (직교 셀에서 정확하다)"
mic(d, h) = (f = h \ d; h * (f .- round.(f)))
"원자 i에서 모든 원자까지의 최소 영상 거리"
distances_from(s::Cell, i) = [norm(mic(s.R[:, j] .- s.R[:, i], s.h)) for j in 1:natoms(s)]

# ---------------------------------------------------------------- PythonCall 다리 (ch10_common.jl 위에)
to_ase(s::Cell; pbc = true) = to_ase(s.Z, s.R, s.h; pbc)
cell_from_ase(a) = Cell(from_ase(a)...)
"네 uMLIP의 ASE 계산기. 모델 이름은 문자열로 받는다. (계산기, torch 모듈)"
make_calc(model::AbstractString; dtype = "float64") = make_calc(Symbol(model); dtype)
n_params(net) = first(torch_param_info(net))

"계산기 호출 횟수를 세는 묶음. 같은 구조를 두 번 넘기면 ASE 계산기가 결과를 재사용하지만, 여기서는 매번 센다"
mutable struct Counted
    calc::Py
    n::Int
end
Counted(calc) = Counted(calc, 0)
"에너지(eV)와 힘(3 x n, eV/Å)"
function energy_forces(c::Counted, s::Cell)
    a = to_ase(s); a.calc = c.calc; c.n += 1
    F = permutedims(pyconvert(Matrix{Float64}, a.get_forces()))
    return pyconvert(Float64, a.get_potential_energy()), F
end
energy(c::Counted, s::Cell) = energy_forces(c, s)[1]

"""ASE에 맡기는 셀 이완: FrechetCellFilter + BFGS, sym이면 FixSymmetry. (구조, 걸음 수, 최대 일반화 힘의 이력)
대칭 제약을 Julia로 다시 쓰지 않은 이유: 공간군 연산을 찾는 spglib 호출과 응력의 대칭화가 함께 필요하다."""
function ase_relax(c::Counted, s::Cell; fmax, sym = false, cell = true, maxsteps = 400)
    a = to_ase(s); a.calc = c.calc
    sym && a.set_constraint(pyimport("ase.constraints").FixSymmetry(a))
    obj = cell ? pyimport("ase.filters").FrechetCellFilter(a) : a
    opt = pyimport("ase.optimize").BFGS(obj; logfile = pybuiltins.None)
    hist = Float64[]
    function logger()                                     # 매 걸음 최대 일반화 힘(원자 힘과 응력 성분)을 기록
        f = pyconvert(Matrix{Float64}, obj.get_forces())
        push!(hist, maximum(sqrt.(sum(abs2, f; dims = 2))))
        return nothing
    end
    opt.attach(logger, interval = 1)                      # Julia 함수는 호출 가능한 파이썬 객체로 넘어간다
    opt.run(fmax = fmax, steps = maxsteps)
    nst = pyconvert(Int, opt.nsteps); c.n += nst + 1
    sg = pyconvert(String, pyimport("ase.spacegroup.symmetrize").check_symmetry(a, 1e-3).international)
    E = pyconvert(Float64, a.get_potential_energy()); σ = pyconvert(Matrix{Float64}, a.get_stress(voigt = false))
    a.set_constraint()
    return (cell = cell_from_ase(a), steps = nst, hist = hist, E = E, sg = sg, stress = σ)
end
"ASE cellpar: 길이 셋과 각 셋(도)"
function cellpar(h)
    L = [norm(h[:, k]) for k in 1:3]
    ang(i, j) = acosd(clamp(dot(h[:, i], h[:, j]) / (L[i] * L[j]), -1, 1))
    return [L; ang(2, 3); ang(1, 3); ang(1, 2)]
end

# ---------------------------------------------------------------- 최적화기 (원자 위치만, ASE 3.29와 같은 식)
"최대 원자 힘 |F_i|"
fmax_of(F) = maximum(sqrt.(vec(sum(abs2, F; dims = 1))))

"""ASE BFGS: H₀ = 70 I (eV/Å²), 걸음 = -V (gᵀV / |ω|), 가장 긴 원자 걸음을 maxstep = 0.2 Å로 자른다.
갱신 H += Δg Δgᵀ / dot(Δx, Δg) - (HΔx)(HΔx)ᵀ / dot(Δx, HΔx). 수렴 판정은 걸음 전에 한다. 힘 호출 수 = 걸음 수 + 1."""
function bfgs(efun, s0::Cell; fmax, maxsteps = 400, alpha = 70.0, maxstep = 0.2)
    s = copy(s0); n = 3 * natoms(s)
    H = Matrix{Float64}(alpha * I, n, n); x0 = Float64[]; g0 = Float64[]
    E, F = efun(s); steps = 0; hist = [fmax_of(F)]
    while hist[end] >= fmax && steps < maxsteps          # ASE: converged는 fmax보다 작을 때
        x = vec(copy(s.R)); g = -vec(F)
        if !isempty(x0)
            dx = x .- x0
            if maximum(abs, dx) >= 1e-7
                dg = g .- g0; a = dot(dx, dg); Hdx = H * dx; b = dot(dx, Hdx)
                H .+= dg * dg' ./ a .- Hdx * Hdx' ./ b
            end
        end
        x0, g0 = x, g
        ω, V = eigen(Symmetric(H))
        dx = -V * ((V' * g) ./ abs.(ω))
        len = sqrt.(vec(sum(abs2, reshape(dx, 3, :); dims = 1)))
        maximum(len) >= maxstep && (dx .*= maxstep / maximum(len))
        s = Cell(s.Z, reshape(x .+ dx, 3, :), s.h)
        E, F = efun(s); steps += 1; push!(hist, fmax_of(F))
    end
    return (cell = s, E = E, F = F, steps = steps, hist = hist)
end

"ASE FIRE (dt 0.1, dtmax 1.0, Nmin 5, finc 1.1, fdec 0.5, a 0.1, fa 0.99, maxstep 0.2, downhill_check 없음)"
function fire(efun, s0::Cell; fmax, maxsteps = 300, dt = 0.1, dtmax = 1.0, Nmin = 5, finc = 1.1, fdec = 0.5,
              astart = 0.1, fa = 0.99, maxstep = 0.2)
    s = copy(s0); E, F = efun(s); steps = 0; v = nothing; a = astart; Nst = 0
    while fmax_of(F) >= fmax && steps < maxsteps
        f = vec(F)
        if v === nothing
            v = zeros(length(f))
        else
            if dot(f, v) > 0
                v = (1 - a) .* v .+ a .* f ./ norm(f) .* norm(v)
                Nst > Nmin && (dt = min(dt * finc, dtmax); a *= fa)
                Nst += 1
            else
                v .= 0; a = astart; dt *= fdec; Nst = 0
            end
        end
        v = v .+ dt .* f
        dr = dt .* v; nd = norm(dr); nd > maxstep && (dr = maxstep .* dr ./ nd)
        s = Cell(s.Z, s.R .+ reshape(dr, 3, :), s.h)
        E, F = efun(s); steps += 1
    end
    return (cell = s, E = E, F = F, steps = steps)
end

# ---------------------------------------------------------------- 상태방정식
"""ASE의 기본 'sj'(안정화된 jellium): E를 x = V^(-1/3)의 3차 다항식으로 맞추고, dE/dx = 0의 근 가운데 곡률이 양인 것.
(V₀, E₀, B [eV/Å³])를 돌려준다."""
function sjeos(V, E)
    x = V .^ (-1 / 3)
    c = [x .^ 3 x .^ 2 x ones(length(x))] \ E           # c₃ x³ + c₂ x² + c₁ x + c₀ (numpy polyfit와 같은 최소제곱)
    d1(t) = 3c[1] * t^2 + 2c[2] * t + c[3]; d2(t) = 6c[1] * t + 2c[2]
    disc = (2c[2])^2 - 12c[1] * c[3]
    disc < 0 && error("No minimum!")
    for t in ((-2c[2] + sqrt(disc)) / (6c[1]), (-2c[2] - sqrt(disc)) / (6c[1]))
        if t > 0 && d2(t) > 0
            return t^-3, c[1] * t^3 + c[2] * t^2 + c[3] * t + c[4], t^5 * d2(t) / 9
        end
    end
    error("No minimum!")
end

# ---------------------------------------------------------------- 유한 변위 포논
"""원시 셀 prim의 rep³ 슈퍼셀에서, 첫 셀의 원자 κ를 ±d만큼 α 방향으로 옮겨(대칭을 쓰지 않아 6n₀번) 힘 상수를 만든다.
힘에서는 알짜 힘(평균)을 뺀다(Lab 10.12의 f - f.mean(0)). 반환: Φ (3, N, 3, n₀) = Φ_{βα}(j; κ), 슈퍼셀, 힘 호출 수.
phonopy는 대칭으로 변위 슈퍼셀을 3개만 만들고 +d만 쓴다. 자리 대칭에 반전이 있으면 -d는 +d와 동등하므로 둘은 같은 중심 차분이다."""
function fd_force_constants(efun, prim::Cell; rep = 2, d = 0.01)
    sc = repeat_cell(prim, rep); n0, N = natoms(prim), natoms(sc)
    Φ = zeros(3, N, 3, n0); calls = 0
    for κ in 1:n0, α in 1:3
        Fpm = map((1, -1)) do sgn
            R = copy(sc.R); R[α, κ] += sgn * d
            _, F = efun(Cell(sc.Z, R, sc.h)); calls += 1
            F .- mean(F; dims = 2)
        end
        Φ[:, :, α, κ] = -(Fpm[1] .- Fpm[2]) ./ (2d)          # Φ_{κα, jβ} = -∂F_{jβ}/∂u_{κα}
    end
    return Φ, sc, calls
end

"""압축된 Φ를 병진으로 펼쳐 전체 (N, N, 3, 3) 행렬 fc[i, j, α, β]를 만들고, phonopy의 symmetrize_force_constants(level = 1)처럼
(병진 불변 → 순열 대칭) 한 번, 그리고 병진 불변을 한 번 더 건다."""
function full_fc(Φ, sc::Cell, prim::Cell, rep)
    n0, N = natoms(prim), natoms(sc)
    cellidx(i) = (c = (i - 1) ÷ n0; ((c ÷ rep^2), (c ÷ rep) % rep, c % rep))
    atom(m, κ) = (mod(m[1], rep) * rep^2 + mod(m[2], rep) * rep + mod(m[3], rep)) * n0 + κ
    fc = zeros(N, N, 3, 3)
    for i in 1:N, j in 1:N
        mi, mj = cellidx(i), cellidx(j); κ = (i - 1) % n0 + 1; λ = (j - 1) % n0 + 1
        jj = atom(mj .- mi, λ)                                   # 원자 i를 첫 셀로 옮겼을 때 j가 가는 자리
        fc[i, j, :, :] = permutedims(Φ[:, jj, :, κ])             # fc[i, j, α, β] = ∂²E/∂u_{iα}∂u_{jβ}
    end
    trans_invariance!(fc); perm_symmetry!(fc); trans_invariance!(fc)
    return fc
end
"phonopy의 set_translational_invariance: 첫 원자 첨자, 둘째 원자 첨자에 대해 차례로 합을 0으로 (오차를 고르게 뺀다)"
function trans_invariance!(fc)
    N = size(fc, 1)
    fc .-= sum(fc; dims = 1) ./ N
    fc .-= sum(fc; dims = 2) ./ N
    return fc
end
"phonopy의 set_permutation_symmetry: Φ_{ij,αβ} ← (Φ_{ij,αβ} + Φ_{ji,βα}) / 2"
function perm_symmetry!(fc)
    c = copy(fc); N = size(fc, 1)
    for i in 1:N, j in 1:N
        fc[i, j, :, :] = (c[i, j, :, :] .+ permutedims(c[j, i, :, :])) ./ 2
    end
    return fc
end

"""공약 q점(분수 좌표, 원시 역격자 기준)의 진동수(cm⁻¹, 오름차순, 허수는 음수).
D_{κα,λβ}(q) = Σ_{j∈λ} fc[κ, j, α, β] exp(i q⋅(r_j - r_κ)) / √(m_κ m_λ). 2×2×2 슈퍼셀에서 Γ, X, M, R은 영상의 선택과 무관하게 정확하다."""
function frequencies(fc, sc::Cell, prim::Cell, q)
    n0, N = natoms(prim), natoms(sc)
    G = 2π * inv(prim.h)'                                       # 역격자 (열 = b_k)
    qc = G * q
    D = zeros(ComplexF64, 3n0, 3n0)
    for κ in 1:n0, j in 1:N
        λ = (j - 1) % n0 + 1
        ph = cis(dot(qc, sc.R[:, j] .- sc.R[:, κ])) / sqrt(MASS[prim.Z[κ]] * MASS[prim.Z[λ]])
        D[3κ-2:3κ, 3λ-2:3λ] .+= fc[κ, j, :, :] .* ph
    end
    w2 = eigvals(Hermitian((D + D') / 2))
    return sort(sign.(w2) .* sqrt.(abs.(w2)) .* SQRT_EV_A2_AMU_CM)
end
const QP = Dict("G" => [0.0, 0, 0], "X" => [0.0, 0.5, 0], "M" => [0.5, 0.5, 0], "R" => [0.5, 0.5, 0.5])
const QORDER = ["G", "X", "M", "R"]

"한 구조, 한 변위의 포논 전체: (진동수 사전, 힘 호출 수)"
function phonons(efun, prim::Cell; rep = 2, d = 0.01)
    Φ, sc, calls = fd_force_constants(efun, prim; rep, d)
    fc = full_fc(Φ, sc, prim, rep)
    return Dict(k => frequencies(fc, sc, prim, QP[k]) for k in QORDER), calls
end

# ---------------------------------------------------------------- 모드 붙이기 (labs/107_compare.py와 같은 규칙)
"정렬한 진동수를 tol(cm⁻¹) 안의 묶음으로: [(평균, 겹침수)]"
function groups(f; tol = 1.5)
    f = sort(f); out = [[f[1]]]
    for x in f[2:end]
        abs(x - out[end][end]) < tol ? push!(out[end], x) : push!(out, [x])
    end
    return [(mean(g), length(g)) for g in out]
end
"Γ: 음향 삼중항을 뺀 삼중항을 오름차순으로 TO1, TO2, t2u, TO3. R: 가장 낮은 삼중항 R4+, 마지막 이중항 R3+, 마지막 홑 R1+"
function assign(fr)
    gG = [g for g in groups(fr["G"]) if !(abs(g[1]) < 3 && g[2] == 3)]
    trip = [g[1] for g in gG if g[2] == 3]
    gR = groups(fr["R"])
    return Dict("G_TO1" => trip[1], "G_TO2" => trip[2], "G_t2u" => trip[3], "G_TO3" => trip[4],
                "R4+" => [g for g in gR if g[2] == 3][1][1], "R3+" => [g for g in gR if g[2] == 2][end][1],
                "R1+" => [g for g in gR if g[2] == 1][end][1], "M3" => minimum(fr["M"]), "X_min" => minimum(fr["X"]))
end
"허수 진동수를 133i처럼"
fmt_w(x) = x < 0 ? @sprintf("%.0fi", abs(x)) : @sprintf("%.0f", x)

