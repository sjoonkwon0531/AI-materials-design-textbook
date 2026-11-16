# lab_7_22.jl — Lab 7.22: 학습된 퍼텐셜로 2 ps NVE 분자동역학 (labs/77_md.py)
# ASE 대신 velocity Verlet을 직접 썼다(Å, fs, amu, eV; 가속도 = F/m · 0.0096485 Å/fs²).
# 시작: 300 K 시험 프레임 2200 + Maxwell–Boltzmann(300 K) 속도, 총운동량·각운동량 0(온도 유지 재척도, ASE와 같은 약속).
# 기준 GFN2-xTB 궤적은 PythonCall로 tblite를 부른다(아래 xtb_ef). 학습된 모형은 lab_7_21.jl이 저장한 매개변수를 읽는다.
include("ch7_common.jl")
include("mlip_common.jl")
include("equiv_layers.jl")
include("models77.jl")
import Serialization
using PythonCall

const CONV = 0.0096485                    # (eV/Å)/amu -> Å/fs²
const KB = 8.617333262e-5                 # eV/K
const MASS = Dict(1 => 1.008, 6 => 12.011, 8 => 15.999)

"MB 속도 → 총운동량 0 → 각운동량 0, 매 단계 뒤 원래 운동 에너지로 재척도 (ASE Stationary/ZeroRotation의 preserve_temperature)"
function init_velocities(X, m, T, rng)
    v = randn(rng, 3, length(m)) .* sqrt.(KB * T * CONV ./ m')
    ek(v) = 0.5 * sum(m' .* v .^ 2) / CONV
    e0 = ek(v)
    v .-= sum(m' .* v; dims = 2) ./ sum(m); v .*= sqrt(e0 / ek(v))
    c = sum(m' .* X; dims = 2) ./ sum(m); r = X .- c
    L = sum(m[i] * cross(r[:, i], v[:, i]) for i in eachindex(m))
    Iten = sum(m[i] * (dot(r[:, i], r[:, i]) * I - r[:, i] * r[:, i]') for i in eachindex(m))
    ω = Iten \ L
    for i in eachindex(m)
        v[:, i] .-= cross(ω, r[:, i])
    end
    v .*= sqrt(e0 / ek(v))
    return v
end

"NVE velocity Verlet, dt 0.5 fs, 4000단계, 5단계마다 기록. 결합 > 2.0 Å 또는 원자쌍 < 0.6 Å이면 불안정"
function run_md(ef, X0, Z, rng; nstep = 4000, every = 5, dt = 0.5)
    m = [MASS[z] for z in Z]; X = copy(X0); v = init_velocities(X, m, 300.0, rng)
    D(X) = [norm(X[:, i] - X[:, j]) for i in 1:9, j in 1:9]
    bonded = 0 .< D(X0) .< 1.7
    E, F = ef(X)
    rec = (Etot = Float64[], T = Float64[], L = Float64[], OH = Float64[], CC = Float64[], CO = Float64[])
    status = "stable"
    for s in 1:nstep
        v .+= 0.5dt .* F ./ m' .* CONV; X .+= dt .* v
        E, F = ef(X); v .+= 0.5dt .* F ./ m' .* CONV
        if s % every == 0
            d = D(X)
            if any(d[bonded] .> 2.0) || any(d[i, j] < 0.6 for i in 1:9 for j in i+1:9)
                status = @sprintf("UNSTABLE at %.2f ps", s * dt * 1e-3); break
            end
            ek = 0.5 * sum(m' .* v .^ 2) / CONV
            push!(rec.Etot, E + ek); push!(rec.T, 2ek / (3 * 9 * KB))
            c = sum(m' .* X; dims = 2) ./ sum(m)
            push!(rec.L, norm(sum(m[i] * cross(X[:, i] - vec(c), v[:, i]) for i in 1:9)))
            push!(rec.OH, d[3, 4]); push!(rec.CC, d[1, 2]); push!(rec.CO, d[2, 3])
        end
    end
    t = (1:length(rec.Etot)) .* every .* dt .* 1e-3                        # ps
    drift = length(t) > 10 ? ([t ones(length(t))] \ rec.Etot)[1] : NaN
    return rec, status, drift
end

"1차원 Wasserstein-1 거리: ∫ |F_u(x) - F_v(x)| dx (경험 CDF)"
function wasserstein1(u, v)
    a = sort(vcat(u, v)); F(w, x) = searchsortedlast(sort(w), x) / length(w)
    return sum(abs(F(u, a[k]) - F(v, a[k])) * (a[k+1] - a[k]) for k in 1:length(a)-1)
end

"tblite GFN2-xTB 에너지(eV)와 힘(eV/Å) — PythonCall (tblite.interface.Calculator는 Bohr, Hartree 단위)"
function xtb_ef_factory(Z)
    np = pyimport("numpy"); tb = pyimport("tblite.interface")
    BOHR, HA = 0.529177210903, 27.211386245988
    function ef(X)
        calc = tb.Calculator("GFN2-xTB", np.array(Z), np.array(permutedims(X) ./ BOHR))
        calc.set("verbosity", 0)
        res = calc.singlepoint()
        E = pyconvert(Float64, res.get("energy")) * HA
        G = pyconvert(Matrix{Float64}, res.get("gradient"))             # (9, 3) Hartree/Bohr
        return E, -permutedims(G) .* (HA / BOHR)
    end
    return ef
end

"학습된 Lux 모형 → X (3 x 9) ↦ (E, F)"
function model_ef(model, ps, st)
    Ef(x) = sum(first(model(reshape(x, 3, 9, 1), ps, st)))
    return X -> (Ef(X), -only(Zygote.gradient(Ef, X)))
end

function main()
    R3, Z, _, _ = load_ethanol(300)
    X0 = R3[:, :, 2201]
    models = Dict{String,Any}("xTB" => xtb_ef_factory(Z))
    for kind in (:mlp, :mlp_aug, :mlp_frame, :schnet, :nequip)
        f = joinpath(@__DIR__, "77_$(kind)_N1000.jls"); isfile(f) || continue
        model = build(kind, Z); _, st = Lux.setup(Xoshiro(0), model)
        ps = to64(open(Serialization.deserialize, f))
        # e0, scale는 lab_7_21.jl과 같은 학습 자료로 다시 맞춘다(Ref라 직렬화에 들어 있지 않다)
        _, _, E3, F3 = load_ethanol(300); tr = read_idx("ethanol_pool_perm.csv")[1:1000]
        model.e0[] = mean(E3[tr]); model.scale[] = sqrt(mean(abs2, F3[:, :, tr]))
        models[string(kind)] = model_ef(model, ps, st)
    end
    ref = nothing
    @printf("%-12s %-22s %12s %10s %8s %6s | %s\n", "model", "status", "drift", "std(Etot)", "max|L|", "<T>", "bond mean (W1 vs xTB) [A]")
    for name in vcat(["xTB"], sort([k for k in keys(models) if k != "xTB"]))
        rec, status, drift = run_md(models[name], X0, Z, Xoshiro(0))
        name == "xTB" && (ref = rec)
        bonds = join([@sprintf("%s %.3f (%.3f)", b, mean(getfield(rec, s)), wasserstein1(getfield(rec, s), getfield(ref, s)))
                      for (b, s) in (("O-H", :OH), ("C-C", :CC), ("C-O", :CO)) if !isempty(getfield(rec, s))], "  ")
        @printf("%-12s %-22s %7.2f meV/ps %6.2f meV %8.4f %4.0f K | %s\n", name, status, 1e3 * drift,
                isempty(rec.Etot) ? NaN : 1e3 * std(rec.Etot; corrected = false), isempty(rec.L) ? NaN : maximum(rec.L),
                isempty(rec.T) ? NaN : mean(rec.T), bonds)
    end
end

main()
