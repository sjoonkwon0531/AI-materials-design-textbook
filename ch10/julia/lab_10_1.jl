# lab_10_1.jl, Lab 10.1: 한 번의 에너지와 힘 계산, 원자 수에 따른 비용 (labs/101_cost.py)
#   julia lab_10_1.jl dft      PySCF 주기 RKS(PBE, gth-szv, gth-pade, Γ점), 다이아몬드 Si N = 2, 8   (PythonCall)
#   julia lab_10_1.jl mlip     MACE-MP-0 small, Si N = 8 … 1000, 그리고 ASE EMT(Cu) N = 108 … 4000  (PythonCall)
#   julia lab_10_1.jl summary  측정값에서 기울기, 두 외삽 모형, 손익분기 S*(식 10.1.1), ROI, 1 nm → 1 μm 비용 (Julia)
# 세 엔진은 모두 파이썬 패키지이므로 측정은 PythonCall로 하고, 비용 모형과 Estimate Box 10.1의 회계는 Julia로 계산한다.
# 시간은 프로세스 CPU 시간(cputime)이다. FROM_PYTHON = true면 파이썬이 잰 data/101_cost.json으로 summary만 계산한다.
include("ch10_common.jl")

const OUT = joinpath(JOUT, "101_cost.json")
const FROM_PYTHON = true
load_res() = isfile(OUT) ? Dict{String,Any}(String(k) => v for (k, v) in pairs(readjson(OUT))) : Dict{String,Any}()

function run_dft(; Ns = (2, 8))
    gto = pyimport("pyscf.pbc.gto"); dft = pyimport("pyscf.pbc.dft")
    res = load_res(); out = Dict{String,Any}()
    for N in Ns
        Z, R, H = N == 2 ? from_ase(pyimport("ase.build").bulk("Si", "diamond"; a = 5.43)) :
                  bulk_cell("Si", "diamond", 5.43)                     # 8원자 입방 셀 (N = 8)
        cell = gto.Cell()
        cell.atom = pylist([("Si", pylist(R[:, k])) for k in axes(R, 2)])
        cell.a = np.array(permutedims(H)); cell.basis = "gth-szv"; cell.pseudo = "gth-pade"; cell.verbose = 0
        cell.build()
        t = cputime()
        mf = dft.RKS(cell); mf.xc = "pbe"; e = pyconvert(Float64, mf.kernel())
        cpu = cputime() - t
        nao = pyconvert(Int, cell.nao_nr())
        out[string(N)] = Dict("E" => e, "cpu" => cpu, "nao" => nao, "conv" => pyconvert(Bool, mf.converged))
        @printf("DFT  N=%3d  nao=%4d  E=%.6f Ha  cpu=%.1f s\n", N, nao, e, cpu)
    end
    res["dft"] = out; writejson(OUT, res)
end

"calc.calculate를 nrep번 부른 평균 CPU 시간. 캐시를 비우려고 calc.results를 매번 지운다(파이썬 판과 같다)."
function time_calls(calc, Z, R, H, nrep)
    a = to_ase(Z, R, H); a.calc = calc
    a.get_forces()                                                      # 예열(그래프 구성, JIT 경로)
    t = cputime()
    for _ in 1:nrep
        calc.results = pydict(); calc.calculate(a, pylist(["energy", "forces"]))
    end
    return (cputime() - t) / nrep
end

function run_mlip()
    calc, _ = make_calc(:mace; dtype = "float64")
    res = load_res(); res["mlip"] = Dict{String,Any}(); rng = Xoshiro(0)
    for rep in 1:5
        Z, R, H = bulk_cell("Si", "diamond", 5.43; n = rep)
        R .+= 0.03 .* randn(rng, size(R))                               # 흔들기 σ = 0.03 Å (Julia 난수)
        cpu = time_calls(calc, Z, R, H, length(Z) <= 512 ? 3 : 1)
        res["mlip"][string(length(Z))] = cpu
        @printf("MACE N=%5d  cpu/call=%.3f s  per atom=%.2f ms\n", length(Z), cpu, 1e3cpu / length(Z))
    end
    emt, _ = make_calc(:emt); res["emt"] = Dict{String,Any}()
    for rep in (3, 5, 8, 10)
        Z, R, H = bulk_cell("Cu", "fcc", 3.61; n = rep)
        cpu = time_calls(emt, Z, R, H, 3)
        res["emt"][string(length(Z))] = cpu
        @printf("EMT  N=%5d  cpu/call=%.4f s  per atom=%.1f us\n", length(Z), cpu, 1e6cpu / length(Z))
    end
    writejson(OUT, res)
end

"log-log 1차 맞춤의 기울기"
loglog_slope(N, t) = slope(log.(N), log.(t))

function summary()
    r = FROM_PYTHON ? readjson(joinpath(DATA, "101_cost.json")) : readjson(OUT)
    sortkeys(d) = sort(parse.(Int, String.(collect(keys(d)))))
    Nd = sortkeys(r["dft"]); td = [Float64(r["dft"][string(n)]["cpu"]) for n in Nd]
    Nm = sortkeys(r["mlip"]); tm = [Float64(r["mlip"][string(n)]) for n in Nm]
    pd = loglog_slope(Nd, td); pm = loglog_slope(Nm[3:end], tm[3:end])
    @printf("measured slope d ln t / d ln N:  DFT (N=%d-%d) %.2f   MACE (N>=%d) %.2f\n", Nd[1], Nd[end], pd, Nm[3], pm)
    # 가장 큰 DFT 측정점(8원자)에 고정한 두 모형: 측정 기울기(낙관적)와 대각화의 점근 법칙 N³(Ch.4 §4.3.1)
    N0, t0 = Nd[end], td[end]
    mace(N) = interp(N, Float64.(Nm), tm)
    for N in (64, 216, 1000)
        tc = t0 * (N / N0)^3; tp = t0 * (N / N0)^pd; tml = mace(N)
        @printf("N=%5d: DFT model  slope %.2f: %9.3g s   cubic: %9.3g s   | MACE measured %7.3f s   ratio %8.3g .. %8.3g\n",
                N, pd, tp, tc, tml, tp / tml, tc / tml)
    end
    c_ml = tm[end] / Nm[end]
    c_emt = mean(Float64(v) / parse(Int, String(k)) for (k, v) in pairs(r["emt"]))
    @printf("per atom-step: MACE %.1f ms, EMT %.0f us, MACE/EMT = %.0f\n", 1e3c_ml, 1e6c_emt, c_ml / c_emt)
    n_si = 8 / 5.43^3                                                   # 다이아몬드 Si의 원자 밀도 (Å⁻³)
    for L in (1, 10, 100, 1000)
        N = n_si * (10L)^3; ts = N * c_ml
        @printf("Si cube %5d nm: N = %9.2e atoms; one MACE step %9.2e s; 1 ns (1e6 steps of 1 fs) %9.2e core-years\n",
                L, N, ts, ts * 1e6 / 3.15e7)
    end
    # Estimate Box 10.1: 라벨은 8원자 셀 400개, 학습 100 s, 생산은 N원자
    n_lab, C_train = 400, 100.0
    for N in (64, 216)
        build = n_lab * t0 + C_train
        Sp = build / (t0 * (N / N0)^pd - mace(N)); Sc = build / (t0 * (N / N0)^3 - mace(N))
        @printf("break-even (N=%d, %d labels of 8 atoms, C_train=%.0f s): build cost %.3g s; S* = %.3g steps (slope model) .. %.3g steps (cubic)\n",
                N, n_lab, C_train, build, Sp, Sc)
    end
    S_md, N_md = 1_000_000, 64
    @printf("ROI: 1 ns (1e6 steps) of %d atoms = %.1e atom-steps; MLIP route %.1e atom-steps per DFT call (%d calls); AIMD route %d atom-steps per DFT call; ratio %.0f\n",
            N_md, S_md * N_md, S_md * N_md / n_lab, n_lab, N_md, S_md / n_lab)
end

"수용장: MACE-MP-0 small의 r_max와 상호작용 층 수 (labs/101_receptive.py)"
function receptive()
    m = pyimport("torch").load(joinpath(DATA, "mace_mp0_small.model"); map_location = "cpu", weights_only = false)
    rc = pyconvert(Float64, m.r_max); T = pyconvert(Int, m.num_interactions)
    @printf("MACE-MP-0 small: r_max = %.1f A, interaction layers = %d, receptive field T*r_max = %.0f A\n", rc, T, T * rc)
end

mode = isempty(ARGS) ? "summary" : ARGS[1]
mode == "dft" ? run_dft() : mode == "mlip" ? run_mlip() : mode == "receptive" ? receptive() : summary()
