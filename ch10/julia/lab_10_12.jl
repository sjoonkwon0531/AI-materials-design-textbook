# lab_10_12.jl: Lab 10.12: 입방정 SrTiO₃의 포논과 팔면체 회전, 네 uMLIP 대 PBE (labs/107_phonons.py, 107_compare.py)
# 실행: julia lab_10_12.jl mace | chgnet | sevennet | mattersim   (모델마다 한 번, 결과는 models/107_<model>.json)
#       julia lab_10_12.jl compare                                (네 결과를 모아 Evarestov 등(2011)의 평면파 PBE와 비교)
# uMLIP의 에너지와 힘은 PythonCall로 ASE 계산기를 부른다. 상태방정식 맞춤, 유한 변위 힘 상수, 동역학 행렬, 모드 붙이기,
# 강체 회전 구조와 회전각은 Julia로 옮겼다(ch10_common2.jl). I4/mcm 대칭을 유지한 셀 + 원자 이완만 ASE(FixSymmetry)에 맡긴다.
# 줄인 것: 그림 10.7.1의 분산 곡선(공약 q점 사이의 보간)은 계산하지 않는다. 필요하면 PythonCall로 phonopy를 부르면 된다.
const ARG = isempty(ARGS) ? "mace" : ARGS[1]
ENV["CH10_MODEL"] = ARG == "compare" ? "mace" : ARG
include("ch10_common2.jl")

const PBE = Dict("G_TO1" => -133, "G_TO2" => 146, "G_t2u" => 226, "G_TO3" => 508, "R4+" => -86, "R3+" => 433, "R1+" => 798)
const KEYS = ["G_TO1", "G_TO2", "G_t2u", "G_TO3", "R4+", "R3+", "R1+", "M3", "X_min"]

"a⁰a⁰c⁻ 강체 회전: 2×2×2 셀에서 적도 산소를 u = (a/2) tan φ만큼, 이웃 셀끼리 반대 방향으로 옮긴다"
function tilted(a, phi_deg)
    s = sto(a; rep = 2); u = 0.5 * a * tand(phi_deg); R = copy(s.R)
    for c in 0:7                                            # 셀 번호 c = 4i + 2j + k (ASE repeat 순서)
        i, j, k = c ÷ 4, (c ÷ 2) % 2, c % 2
        sg = (-1)^(i + j + k)
        R[2, 5c+5] += -sg * u                               # Ti - (a/2)x 자리의 O: y 방향으로
        R[1, 5c+4] += sg * u                                # Ti - (a/2)y 자리의 O: x 방향으로
    end
    return Cell(s.Z, R, s.h)
end

"적도 산소의 Ti–O–Ti 각에서 평균 회전각: φ = (180° - 각) / 2"
function tilt_angle(s::Cell)
    ti = [5c + 2 for c in 0:7]; ang = Float64[]
    for c in 0:7, o in (5c + 4, 5c + 5)
        d = [mic(s.R[:, t] .- s.R[:, o], s.h) for t in ti]
        n = sortperm(norm.(d))[1:2]; v1, v2 = d[n[1]], d[n[2]]
        push!(ang, acosd(clamp(dot(v1, v2) / (norm(v1) * norm(v2)), -1, 1)))
    end
    return (180 - mean(ang)) / 2
end

function run_model(model)
    dtype = model == "mace" ? "float64" : "float32"
    calc, net = make_calc(model; dtype); c = Counted(calc)
    np_ = n_params(net)
    @printf("=== %s  (%s parameters, %s) ===\n", LABEL[model], string(np_), dtype)
    efun(s) = energy_forces(c, s)
    t_start = time()

    # 1. 상태방정식: a = 3.86–4.02 Å의 9점, Birch–Murnaghan
    avals = range(3.86, 4.02; length = 9)
    E = [energy(c, sto(a)) for a in avals]
    v0, _, B, _ = birch_murnaghan(collect(avals) .^ 3, E)        # ch10_common.jl: V^(-2/3)의 3차식, ASE와 같은 최적점
    a0 = cbrt(v0)
    @printf("  EOS: a0 = %.4f A, B = %.1f GPa   (exp a = 3.905 A; PBE/VASP 3.944 A)\n", a0, B * GPA)

    # 2. 포논: 세 격자상수, 2×2×2, 변위 0.01 Å
    ph = Dict{String,Any}()
    for (tag, a) in (("own", a0), ("pbe", 3.944), ("exp", 3.905))
        t0 = time(); fr, calls = phonons(efun, sto(a); rep = 2, d = 0.01)
        ph[tag] = Dict("a" => a, "ncalls" => calls, "seconds" => time() - t0, "freqs" => fr)
        @printf("  phonons at a = %.4f A (%s), %d displaced 40-atom supercells, %.1f s\n", a, tag, calls, time() - t0)
        for k in QORDER
            println("     ", k, ": ", join(fmt_w.(fr[k][1:6]), " "), @sprintf("  ... max %.0f cm-1", fr[k][end]))
        end
    end

    # 3. R점 회전: 강체 회전 곡선(0–10°, 0.5° 간격)과 I4/mcm 이완(4°에서 출발)
    Ecub8 = energy(c, sto(a0; rep = 2))
    phis = collect(0:0.5:10)
    prof = [(energy(c, tilted(a0, p)) - Ecub8) / 8 * 1000 for p in phis]
    imin = argmin(prof)
    @printf("  rigid a0a0c- tilt at a0: min %.2f meV/f.u. at %.1f deg; E(10 deg) = %+.1f meV/f.u.\n", prof[imin], phis[imin], prof[end])
    rel = ase_relax(c, tilted(a0, 4.0); fmax = 0.002, sym = true, maxsteps = 300)
    phi_rel = tilt_angle(rel.cell); dE_rel = (rel.E - Ecub8) / 8 * 1000; cp = cellpar(rel.cell.h)
    @printf("  relaxed tilt (%s, %d steps): phi = %.2f deg, dE = %.2f meV/f.u., cell %.4f %.4f %.4f A, c/a(pseudo) = %.5f\n",
            rel.sg, rel.steps, phi_rel, dE_rel, cp[1], cp[2], cp[3], cp[3] / cp[1])

    # 4. 40원자 셀 한 번의 시간 (벽시계, 다섯 번의 중앙값)
    s = sto(a0; rep = 2); s.R .+= 0.01 .* randn(Xoshiro(0), size(s.R))
    ts = Float64[]
    for _ in 1:5
        s.R[1, 1] += 1e-4; push!(ts, @elapsed energy_forces(c, s))
    end
    @printf("  wall time per 40-atom energy+force call (median of 5): %.0f ms; force calls of this script %d; total %.0f s\n",
            1000 * median(ts), c.n, time() - t_start)
    res = Dict("model" => model, "n_params" => np_, "eos" => Dict("a0" => a0, "B" => B * GPA), "ph" => ph,
               "tilt" => Dict("phis" => phis, "prof" => prof, "phi_rel" => phi_rel, "dE_rel" => dE_rel, "steps" => rel.steps,
                              "cell" => cp), "t_call" => median(ts), "calls" => c.n)
    writejson(joinpath(JOUT, "107_$(model).json"), res)
end

"네 결과를 모아 비교한다. FROM_PYTHON = true이면 파이썬 판의 data/107_<model>.json을 읽는다(그 파일의 키 이름도 같다)"
function compare(; from_python = false)
    D = Dict(m => readjson(joinpath(from_python ? DATA : JOUT, "107_$(m).json")) for m in MODELS)
    fr(m, tag) = Dict(String(k) => Float64.(v) for (k, v) in D[m]["ph"][tag]["freqs"])
    println("=== lattice constant, bulk modulus ===")
    for m in MODELS
        a0 = D[m]["eos"]["a0"]
        @printf("  %-16s a0 = %.4f A (%+.2f%% vs exp 3.905), B = %.0f GPa\n", LABEL[m], a0, 100 * (a0 / 3.905 - 1), D[m]["eos"]["B"])
    end
    a0s = [D[m]["eos"]["a0"] for m in MODELS]
    @printf("  spread of a0: %.4f-%.4f A (PBE/VASP 3.944)\n", minimum(a0s), maximum(a0s))
    for (tag, title) in (("pbe", "at the PBE lattice constant a = 3.944 A"), ("own", "at each model's own a0"), ("exp", "at the experimental a = 3.905 A"))
        println("\n=== frequencies (cm-1) ", title, " ===")
        println("  ", rpad("", 16), join(lpad.(KEYS, 8)))
        tag == "pbe" && println("  ", rpad("PBE (Evarestov)", 16), join(lpad.([haskey(PBE, k) ? fmt_w(PBE[k]) : "-" for k in KEYS], 8)))
        A = Dict(m => assign(fr(m, tag)) for m in MODELS)
        for m in MODELS
            println("  ", rpad(LABEL[m], 16), join(lpad.([fmt_w(A[m][k]) for k in KEYS], 8)))
        end
        tag == "pbe" || continue
        println("  stable-mode ratio w(MLIP)/w(PBE) for G_TO2, G_t2u, G_TO3, R3+, R1+ and its mean; omega^2 ratio of soft modes:")
        for m in MODELS
            rr = [A[m][k] / PBE[k] for k in ("G_TO2", "G_t2u", "G_TO3", "R3+", "R1+")]
            s2 = [(A[m][k] / PBE[k])^2 * sign(A[m][k]) * sign(PBE[k]) for k in ("G_TO1", "R4+")]
            @printf("    %-16s %s  mean %.3f | w^2 ratio G_TO1 %.2f, R4+ %.2f\n", LABEL[m], join([@sprintf("%.3f", x) for x in rr], " "), mean(rr), s2[1], s2[2])
        end
    end
    println("\n=== a0a0c- tilt (2x2x2, I4/mcm relaxation of cell and positions) ===")
    println("  PBE/VASP (Evarestov): 4.9 deg; experiment 2.1 deg (4 K)")
    for m in MODELS
        t = D[m]["tilt"]; prof = Float64.(t["prof"]); i = argmin(prof)
        @printf("  %-16s relaxed tilt %.2f deg, well depth %.2f meV/f.u., c/a %.4f; rigid-tilt minimum %.2f meV/f.u. at %.1f deg, E(10 deg) %+.1f\n",
                LABEL[m], t["phi_rel"], t["dE_rel"], t["cell"][3] / t["cell"][1], prof[i], t["phis"][i], prof[end])
    end
end

ARG == "compare" ? compare(; from_python = get(ENV, "FROM_PYTHON", "false") == "true") : run_model(ARG)
