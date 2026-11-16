# dscribe_bridge.jl — SOAP·MBTR·EMT처럼 Julia에 성숙한 구현이 없는 계산을 PythonCall.jl로 부르는 다리
# descriptors.jl을 먼저 include해야 한다(Structure 정의).
# ---------------------------------------------------------------- DScribe 다리 (SOAP, MBTR)
# SOAP·MBTR은 성숙한 Julia 구현이 없으므로 PythonCall.jl로 DScribe를 부른다.
# 필요한 것: ] add PythonCall, 그리고 PythonCall이 쓰는 파이썬 환경에 ase, dscribe 설치
# (CondaPkg.jl을 쓰면 julia> using CondaPkg; CondaPkg.add_pip("dscribe"); CondaPkg.add("ase")).
using PythonCall
const ase = pyimport("ase")
const dscribe_desc = pyimport("dscribe.descriptors")
const np = pyimport("numpy")

"Julia Structure -> ase.Atoms (전치에 주의)"
function to_ase(s::Structure)
    pos = np.array(permutedims(s.R))
    if s.cell === nothing
        return ase.Atoms(; numbers = np.array(s.Z), positions = pos)
    else
        return ase.Atoms(; numbers = np.array(s.Z), positions = pos, cell = np.array(permutedims(s.cell)), pbc = true)
    end
end

"ase.Atoms -> Julia Structure"
function from_ase(a)
    Z = pyconvert(Vector{Int}, a.numbers)
    R = permutedims(pyconvert(Matrix{Float64}, a.positions))
    periodic = pyconvert(Bool, a.pbc.any())
    return Structure(Z, R, periodic ? permutedims(pyconvert(Matrix{Float64}, np.array(a.cell))) : nothing)
end

"DScribe 기술자 객체의 create 결과를 Julia 배열로"
dscribe_create(desc, s::Structure; kw...) = pyconvert(Array{Float64}, desc.create(to_ase(s); kw...))

"ASE 3.26 이상과 DScribe 2.1.2의 호환 shim(labs/62_dscribe_compat.py)을 먼저 불러 둔다"
function load_dscribe_shim(labs)
    sys = pyimport("sys"); sys.path.insert(0, labs)
    pyimport("importlib").import_module("62_dscribe_compat")
    return nothing
end

"ASE EMT 퍼텐셜의 원자당 에너지(eV). Julia에는 ASE EMT와 같은 매개변수의 구현이 없으므로 그대로 부른다."
function emt_energy_per_atom(s::Structure)
    a = to_ase(s); a.calc = pyimport("ase.calculators.emt").EMT()
    return pyconvert(Float64, a.get_potential_energy()) / natoms(s)
end
