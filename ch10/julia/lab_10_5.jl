# lab_10_5.jl, Lab 10.5: 네 범용 MLIP를 같은 시험대에 올리기 (labs/103_bench.py, 103_inspect.py, 103_stresscheck.py,
#               103_c44.py, 103_nve_dt.py, 103_merge.py)
#   julia lab_10_5.jl <model> [part ...]   model ∈ mace, chgnet, sevennet, mattersim
#       parts: info eos elastic stresscheck c44 timing equiv nve nve_dt ood  (생략하면 전부)
#   julia lab_10_5.jl merge                네 모델의 결과를 표로 (FROM_PYTHON = true면 파이썬의 data/103_<model>.json)
# 네 모델은 PythonCall로 부르는 ASE 계산기다. 한 Julia 세션은 파이썬 하나만 쓰므로 모델에 맞는 가상환경을 골라 실행한다:
#   JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/tmp/claude-0/v10/bin/python  julia lab_10_5.jl mace      (chgnet도 같다)
#   JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/tmp/claude-0/v10b/bin/python julia lab_10_5.jl sevennet  (mattersim도 같다)
# Julia로 직접 계산하는 것: Birch–Murnaghan 맞춤, 변형-응력의 중심 차분, 탄성 상수와 B의 조립, 속도 Verlet NVE와 표류 맞춤,
# 무작위 회전, 이합체 곡선의 판정. 원자 이완(탄성 상수의 relaxed-ion)은 ASE의 BFGS를 PythonCall로 부른다.
include("ch10_common.jl")

const ELEMS = ["Al", "Cu", "Ag", "Au", "Si", "Ge", "Mo", "W"]
const FROM_PYTHON = true
const LABEL = Dict("mace" => "MACE-MP-0 (small)", "chgnet" => "CHGNet 0.3.0", "sevennet" => "SevenNet-0", "mattersim" => "MatterSim-v1.0.0-1M")
const SHORT = Dict("mace" => "MACE", "chgnet" => "CHGNet", "sevennet" => "7net-0", "mattersim" => "MatSim")
resfile(model) = joinpath(JOUT, "103_$(model).json")
function save_part(model, part, payload)
    d = isfile(resfile(model)) ? Dict{String,Any}(String(k) => v for (k, v) in pairs(readjson(resfile(model)))) : Dict{String,Any}()
    d[part] = payload; writejson(resfile(model), d)
end

"ASE Atoms 하나를 셀만 비례해 바꾼 사본 (scale_atoms = true)"
function scaled_copy(ref, F)
    a = ref.copy(); a.set_cell(np.array(jarr(Float64, ref.cell.array) * F); scale_atoms = true); return a
end

# ---------------------------------------------------------------- B. Delta 규약의 상태 방정식
function part_eos(model, calc)
    dcdft = pyimport("ase.collections").dcdft; out = Dict{String,Any}(); t0 = cputime()
    for el in ELEMS
        ref = dcdft[el]; d = dcdft.data[el]; nat = pyconvert(Int, pybuiltins.len(ref))
        v_pbe = pyconvert(Float64, d["wien2k_volume"]); B_pbe = pyconvert(Float64, d["wien2k_B"])
        s = v_pbe * nat / pyconvert(Float64, ref.get_volume())
        vols, ens = Float64[], Float64[]
        for f in range(0.94, 1.06; length = 7)                        # WIEN2k 평형 부피의 94–106 %, 7점
            a = scaled_copy(ref, (s * f)^(1 / 3) * I3); a.calc = calc
            push!(vols, pyconvert(Float64, a.get_volume()) / nat); push!(ens, pyconvert(Float64, a.get_potential_energy()) / nat)
        end
        V0, _, B0, _ = birch_murnaghan(vols, ens)
        out[el] = Dict("V0" => V0, "B0" => B0 * GPA, "V0_pbe" => v_pbe, "B0_pbe" => B_pbe,
                       "V0_exp" => pyconvert(Float64, d["exp_volume"]), "B0_exp" => pyconvert(Float64, d["exp_B"]))
        @printf("[%s] EOS %-2s: V0 = %7.3f A^3/atom (PBE %7.3f)  B0 = %6.1f GPa (PBE %6.1f)\n", model, el, V0, v_pbe, B0 * GPA, B_pbe)
    end
    out["cpu_s"] = cputime() - t0
    save_part(model, "eos", out)
end
eos_a0(model, sym, nat) = (Float64(readjson(resfile(model))["eos"][sym]["V0"]) * nat)^(1 / 3)

# ---------------------------------------------------------------- C. 탄성 상수 (응력-변형, Si는 원자 이완)
"변형 F = I + ε를 준 입방 셀의 Voigt 응력(GPa). relax면 ASE BFGS(fmax = fmax)로 원자 위치를 이완한다."
function strained_stress(calc, base, e; relax = false, fmax = 1e-3, steps = 200)
    a = scaled_copy(base, I3 + e); a.calc = calc
    relax && pyimport("ase.optimize").BFGS(a; logfile = nothing).run(; fmax, steps)
    return jarr(Float64, a.get_stress(; voigt = true)) .* GPA
end
function cubic_elastic(calc, sym, crystal, a0; relax, eps)
    base = pyimport("ase.build").bulk(sym, crystal; a = a0, cubic = true)
    sp = strained_stress(calc, base, Diagonal([eps, 0, 0]); relax); sm = strained_stress(calc, base, Diagonal([-eps, 0, 0]); relax)
    C11 = (sp[1] - sm[1]) / (2eps); C12 = ((sp[2] - sm[2]) + (sp[3] - sm[3])) / 2 / (2eps)
    e = zeros(3, 3); e[2, 3] = e[3, 2] = eps / 2                       # 공학 전단 γ_yz = 2 e_yz = eps
    sp = strained_stress(calc, base, e; relax); sm = strained_stress(calc, base, -e; relax)
    return Dict("C11" => C11, "C12" => C12, "C44" => (sp[4] - sm[4]) / (2eps))
end
function part_elastic(model, calc)
    out = Dict{String,Any}(); t0 = cputime()
    for (sym, crystal, nat) in (("Si", "diamond", 8), ("Cu", "fcc", 4)), eps in (0.005, 0.01)
        a0 = eos_a0(model, sym, nat)
        r = cubic_elastic(calc, sym, crystal, a0; relax = crystal == "diamond", eps)
        r["B_from_C"] = (r["C11"] + 2r["C12"]) / 3; r["a0"] = a0
        r["B_eos"] = Float64(readjson(resfile(model))["eos"][sym]["B0"])
        out["$(sym)_$eps"] = r
        @printf("[%s] elastic %s (strain %.1f%%): C11 = %6.1f, C12 = %6.1f, C44 = %6.1f GPa; (C11+2C12)/3 = %6.1f vs B_EOS = %6.1f\n",
                model, sym, 100eps, r["C11"], r["C12"], r["C44"], r["B_from_C"], r["B_eos"])
    end
    out["cpu_s"] = cputime() - t0
    save_part(model, "elastic", out)
end

# ---------------------------------------------------------------- D. 응력은 에너지의 변형 미분인가, B는 변형 크기에 의존하는가
function part_stresscheck(model, calc)
    out = Dict{String,Any}()
    for (sym, crystal, nat) in (("Si", "diamond", 8), ("Cu", "fcc", 4))
        base = pyimport("ase.build").bulk(sym, crystal; a = eos_a0(model, sym, nat), cubic = true)
        V = pyconvert(Float64, base.get_volume()); h = 1e-3
        Eof(F) = (a = scaled_copy(base, F); a.calc = calc; pyconvert(Float64, a.get_potential_energy()))
        Sof(F) = (a = scaled_copy(base, F); a.calc = calc; jarr(Float64, a.get_stress(; voigt = false)))
        s_calc = Sof(I3)[1, 1]
        s_fd = (Eof(I3 + Diagonal([h, 0, 0])) - Eof(I3 - Diagonal([h, 0, 0]))) / (2h) / V
        r = Dict{String,Any}("stress_xx_calc_GPa" => s_calc * GPA, "stress_xx_fd_GPa" => s_fd * GPA)
        for x in (0.005, 0.01)
            E = [Eof((1 + s) * I3) for s in (-x, 0.0, x)]
            r["B_E_$x"] = (E[1] - 2E[2] + E[3]) / x^2 / (9V) * GPA        # d²E/dε² = 9 V B (등방 변형)
            P = [-tr(Sof((1 + s) * I3)) / 3 for s in (-x, x)]
            r["B_S_$x"] = -(P[2] - P[1]) / (3 * 2x) * GPA
        end
        out[sym] = r
        @printf("[%s] %s: sigma_xx calc = %+.3f GPa, FD of E = %+.3f GPa; B from E'' (0.5%%/1%%) = %.1f/%.1f GPa\n",
                model, sym, r["stress_xx_calc_GPa"], r["stress_xx_fd_GPa"], r["B_E_0.005"], r["B_E_0.01"])
    end
    save_part(model, "stresscheck", out)
end

"Si C44: 원자 고정 대 원자 이완 (전단 1 %, 내부 이완의 크기)"
function part_c44(model, calc)
    base = pyimport("ase.build").bulk("Si", "diamond"; a = eos_a0(model, "Si", 8), cubic = true); g = 0.01
    out = Dict{String,Any}()
    for relax in (false, true)
        s = Float64[]
        for sgn in (1, -1)
            e = zeros(3, 3); e[2, 3] = e[3, 2] = sgn * g / 2
            a = scaled_copy(base, I3 + e); a.calc = calc; x0 = jarr(Float64, a.positions)
            relax && pyimport("ase.optimize").BFGS(a; logfile = nothing).run(; fmax = 1e-4, steps = 500)
            push!(s, jarr(Float64, a.get_stress(; voigt = true))[4] * GPA)
            relax && sgn == 1 && (out["max_shift_A"] = maximum(abs.(jarr(Float64, a.positions) .- x0)))
        end
        out[relax ? "C44_relaxed" : "C44_clamped"] = (s[1] - s[2]) / (2g)
    end
    @printf("[%s] Si C44 clamped = %.1f GPa, relaxed = %.1f GPa, ratio %.2f, max ionic shift %.4f A at 1 %% shear\n",
            model, out["C44_clamped"], out["C44_relaxed"], out["C44_relaxed"] / out["C44_clamped"], out["max_shift_A"])
    save_part(model, "c44", out)
end

# ---------------------------------------------------------------- E. 원자당 CPU 시간
function part_timing(model, calc)
    out = Dict{String,Any}()
    for n in 1:3                                                       # 8, 64, 216원자 (512원자는 공유 메모리 한도를 넘었다)
        Z, R, H = bulk_cell("Si", "diamond", 5.47; n)
        R .+= 0.02 .* randn(Xoshiro(1), size(R))
        a = to_ase(Z, R, H); a.calc = calc; a.get_forces()            # 예열
        ts = map(1:3) do _
            a.positions = np.array(jarr(Float64, a.positions) .+ 1e-4)   # 캐시 무효화
            t0 = cputime(); a.get_potential_energy(); a.get_forces(); cputime() - t0
        end
        t = median(ts); out[string(length(Z))] = Dict("cpu_s" => t, "us_per_atom" => 1e6t / length(Z))
        @printf("[%s] timing N=%4d: %7.3f s CPU per energy+force call, %6.2f ms/atom\n", model, length(Z), t, 1e3t / length(Z))
    end
    save_part(model, "timing", out)
end

# ---------------------------------------------------------------- F. 회전 등변성
function part_equiv(model, calc, dtype)
    Z, R, H = bulk_cell("NaCl", "rocksalt", 5.64; n = 2)              # 64원자, 이온 크기가 다른 두 원소
    R .+= 0.1 .* randn(Xoshiro(3), size(R))
    E0, F0, S0 = calc_efs(calc, Z, R, H; stress = true)
    rng = Xoshiro(10); dE, dF, dS = Float64[], Float64[], Float64[]
    for _ in 1:5
        Q = random_rotation(rng)                                       # 셀과 좌표를 함께 돌린다
        E, F, S = calc_efs(calc, Z, Q * R, Q * H; stress = true)
        push!(dE, abs(E - E0) / length(Z)); push!(dF, maximum(abs.(F .- Q * F0))); push!(dS, maximum(abs.(S .- Q * S0 * Q')) * GPA)
    end
    out = Dict("dE_per_atom" => maximum(dE), "dF_max" => maximum(dF), "dS_max_GPa" => maximum(dS),
               "F_rms" => sqrt(mean(abs2, F0)), "dtype" => dtype)
    @printf("[%s] equivariance (NaCl 64 at, 5 random rotations): max|dE|/N = %.2e eV, max|F(Qx)-QF(x)| = %.2e eV/A, max|dsigma| = %.2e GPa\n",
            model, out["dE_per_atom"], out["dF_max"], out["dS_max_GPa"])
    save_part(model, "equiv", out)
end

# ---------------------------------------------------------------- G. NVE (Julia 속도 Verlet, 힘은 PythonCall)
"""Si 64원자, 1200 K에서 시작, Δt, 걸음 수. 초기 속도는 ASE의 MaxwellBoltzmannDistribution(rng = default_rng(7))과 Stationary를
PythonCall로 불러 파이썬 판과 같은 시작점을 쓴다. 적분은 Julia의 velocity_verlet이다."""
function nve_run(calc, dt, nsteps)
    md = pyimport("ase.md.velocitydistribution")
    a = to_ase(bulk_cell("Si", "diamond", 5.47; n = 2)...)
    md.MaxwellBoltzmannDistribution(a; temperature_K = 1200, rng = np.random.default_rng(7)); md.Stationary(a)
    Z, R0, H = from_ase(a)
    v0 = permutedims(jarr(Float64, a.get_velocities())) .* sqrt(1 / AMU)   # ASE 속도 단위 √(eV/amu) → Å/fs
    mass = MSI * AMU; n = length(Z)
    force(R) = (r = calc_efs(calc, Z, R, H); (r[1], r[2]))
    E, Tk = Float64[], Float64[]
    rec(s, R, v, e) = (push!(E, (e + kinetic(v, mass)) / n); push!(Tk, temperature(v, mass) * (3n - 3) / (3n)))  # ASE: 3N 자유도
    c0 = cputime(); velocity_verlet(force, R0, v0, mass, dt, nsteps; record = rec)
    return E, Tk, cputime() - c0
end
function part_nve(model, calc)
    dt, nsteps = 2.0, 400
    E, Tk, cpu = nve_run(calc, dt, nsteps)
    t = (0:length(E)-1) .* dt ./ 1000; k = length(E) ÷ 4 + 1                # 처음 0.2 ps(등분배)를 버린다
    out = Dict("dt_fs" => dt, "nsteps" => nsteps, "drift_meV_atom_ps" => 1e3 * slope(t[k:end], E[k:end]),
               "fluct_meV_atom" => 1e3 * std(E[k:end]; corrected = false), "T_mean" => mean(Tk[k:end]), "cpu_s" => cpu,
               "E" => 1e3 .* (E .- E[1]))
    @printf("[%s] NVE Si64, dt %.1f fs, %d steps: <T> = %.0f K, drift = %+.3f meV/atom/ps, std(E_tot) = %.3f meV/atom, CPU %.0f s\n",
            model, dt, nsteps, out["T_mean"], out["drift_meV_atom_ps"], out["fluct_meV_atom"], cpu)
    save_part(model, "nve", out)
end
function part_nve_dt(model, calc)
    out = Dict{String,Any}()
    for dt in (1.0, 2.0)
        n = round(Int, 400 / dt); E, _, cpu = nve_run(calc, dt, n)
        t = (0:length(E)-1) .* dt ./ 1000; k = length(E) ÷ 2 + 1
        r = Dict("drift" => 1e3 * slope(t[k:end], E[k:end]), "std" => 1e3 * std(E[k:end]; corrected = false), "dE_end" => 1e3 * (E[end] - E[1]))
        out[string(dt)] = r
        @printf("[%s] dt = %.1f fs, %d steps (0.4 ps): drift (2nd half) = %+.3f meV/atom/ps, std = %.3f meV/atom, E(end)-E(0) = %+.3f meV/atom, CPU %.0f s\n",
                model, dt, n, r["drift"], r["std"], r["dE_end"], cpu)
    end
    save_part(model, "nve_dt", out)
end

# ---------------------------------------------------------------- H. 학습 분포 밖의 두 탐침
"이합체 곡선의 판정: 최소 위치, 최솟값, 1.0 Å의 값, 최소점 안쪽의 단조성 (4 Å 값을 0으로)"
function dimer_stats(rs, Es)
    Es = Es .- Es[end]; i = argmin(Es)
    return Dict("r" => rs, "E" => Es, "r_min" => rs[i], "E_min" => Es[i], "monotone_inside" => all(diff(Es[1:i]) .< 0),
                "E_at_1A" => Es[argmin(abs.(rs .- 1.0))])
end
function part_ood(model, calc)
    out = Dict{String,Any}(); rs = round.(collect(0.6:0.1:4.0); digits = 2)
    for el in ("Si", "Cu")
        z = el == "Si" ? 14 : 29
        Es = [calc_energy(calc, [z, z], [0.0 r; 0.0 0.0; 0.0 0.0], 18.0 * I3) for r in rs]
        d = dimer_stats(rs, Es); out[el*"2"] = d
        @printf("[%s] dimer %s2: r_min = %.1f A, E_min = %.2f eV, E(1.0 A) = %+.2f eV, repulsive wall monotone below r_min: %s\n",
                model, el, d["r_min"], d["E_min"], d["E_at_1A"], d["monotone_inside"])
    end
    dcdft = pyimport("ase.collections").dcdft; ref = dcdft["Si"]; d = dcdft.data["Si"]
    nat = pyconvert(Int, pybuiltins.len(ref))
    s = pyconvert(Float64, d["wien2k_volume"]) * nat / pyconvert(Float64, ref.get_volume()); fr = [1.0, 0.9, 0.8, 0.7, 0.6]
    P = map(fr) do f
        a = scaled_copy(ref, (s * f)^(1 / 3) * I3); a.calc = calc
        -tr(jarr(Float64, a.get_stress(; voigt = false))) / 3 * GPA
    end
    out["Si_compress"] = Dict("V_over_V0" => fr, "P_GPa" => P)
    println("[$model] Si diamond compressed: ", join([@sprintf("V/V0=%.1f: P=%6.1f GPa", f, p) for (f, p) in zip(fr, P)], ", "))
    save_part(model, "ood", out)
end

# ---------------------------------------------------------------- 표 (labs/103_merge.py의 줄임판)
function merge_tables()
    models = ["mace", "chgnet", "sevennet", "mattersim"]
    R = Dict(m => readjson(FROM_PYTHON ? joinpath(DATA, "103_$m.json") : resfile(m)) for m in models)
    println("=== B. EOS: dV0 (%) relative to PBE (WIEN2k) ===")
    dv = Dict(m => [100 * (R[m]["eos"][el]["V0"] / R[m]["eos"][el]["V0_pbe"] - 1) for el in ELEMS] for m in models)
    db = Dict(m => [R[m]["eos"][el]["B0"] / R[m]["eos"][el]["B0_pbe"] for el in ELEMS] for m in models)
    for (k, el) in enumerate(ELEMS)
        e = R["mace"]["eos"][el]
        @printf("  %-4s |%s|  %+7.2f\n", el, join([@sprintf("%+9.2f", dv[m][k]) for m in models]), 100 * (e["V0_exp"] / e["V0_pbe"] - 1))
    end
    @printf("  MAE  |%s|  %7.2f\n", join([@sprintf("%9.2f", mean(abs.(dv[m]))) for m in models]),
            mean(abs(100 * (R["mace"]["eos"][el]["V0_exp"] / R["mace"]["eos"][el]["V0_pbe"] - 1)) for el in ELEMS))
    @printf("  mean B0(model)/B0(PBE) |%s\n", join([@sprintf("%9.3f", mean(db[m])) for m in models]))
    println("  elements with B0(model) < B0(PBE): ", Dict(SHORT[m] => count(<(1), db[m]) for m in models), " of ", length(ELEMS))
    println("=== C. Si elastic constants (GPa), strain 0.5 % / 1.0 % ===")
    for m in models
        a, b = R[m]["elastic"]["Si_0.005"], R[m]["elastic"]["Si_0.01"]
        @printf("   %-7s C11 %6.1f/%6.1f  C12 %6.1f/%6.1f  C44 %6.1f/%6.1f   (C11+2C12)/3 = %6.1f  B_EOS = %6.1f\n", SHORT[m],
                a["C11"], b["C11"], a["C12"], b["C12"], a["C44"], b["C44"], (a["C11"] + 2a["C12"]) / 3, a["B_eos"])
    end
    println("=== G. NVE, Si 64 atoms, MB 1200 K start, dt 2 fs, 400 steps ===")
    for m in models
        E = Float64.(R[m]["nve"]["E"]) ./ 1e3; t = (0:length(E)-1) .* 2.0 ./ 1000; k = length(E) ÷ 4 + 1
        @printf("   %-7s drift = %+7.3f meV/atom/ps  std(E) = %.3f meV/atom   (refit from the stored trace)\n", SHORT[m],
                1e3 * slope(t[k:end], E[k:end]), 1e3 * std(E[k:end]; corrected = false))
    end
    println("=== H. OOD probes (refit from the stored dimer curves) ===")
    for m in models, el in ("Si2", "Cu2")
        o = R[m]["ood"][el]; d = dimer_stats(Float64.(o["r"]), Float64.(o["E"]))
        @printf("   %-7s %s: r_min %.1f A, E_min %+.2f eV, E(1.0 A) %+.2f eV, monotone %s\n", SHORT[m], el, d["r_min"], d["E_min"], d["E_at_1A"], d["monotone_inside"])
    end
end

function main(args)
    isempty(args) && error("usage: julia lab_10_5.jl <model> [parts...] | merge")
    args[1] == "merge" && return merge_tables()
    model = args[1]; parts = length(args) > 1 ? args[2:end] :
        ["info", "eos", "elastic", "stresscheck", "c44", "timing", "equiv", "nve", "nve_dt", "ood"]
    calc, net = make_calc(Symbol(model))
    n, dtype = torch_param_info(net)
    if "info" in parts
        @printf("[%s] %s: params %d, %s\n", model, LABEL[model], n, dtype)
        save_part(model, "info", Dict("label" => LABEL[model], "n_params" => n, "dtype" => dtype))
    end
    "eos" in parts && part_eos(model, calc)
    "elastic" in parts && part_elastic(model, calc)
    "stresscheck" in parts && part_stresscheck(model, calc)
    "c44" in parts && part_c44(model, calc)
    "timing" in parts && part_timing(model, calc)
    "equiv" in parts && part_equiv(model, calc, dtype)
    "nve" in parts && part_nve(model, calc)
    "nve_dt" in parts && part_nve_dt(model, calc)
    "ood" in parts && part_ood(model, calc)
end

main(ARGS)
