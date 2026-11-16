# lab_10_8.jl: Lab 10.8: 세 학생의 성적표 (labs/104_val.py)
# 실행: julia lab_10_8.jl      (lab_10_7.jl을 시드 0으로 먼저 돌려 models/104_al_{qbc,rand,aimd}.jls가 있어야 한다)
# 오라클(MACE-MP-0 small, float32)만 PythonCall로 부른다. 학생 SchNet과 그 학습(ch10_common.jl), 누설 실험의 분할,
# 다섯 조건의 힘 RMSE, 상태방정식(ASE 기본 'sj' 식), Γ점 광학 포논(부격자 강체 이동의 중심 차분), 빈자리 이완(ASE와 같은 FIRE),
# BAOAB Langevin MD, 동경 분포 함수는 모두 Julia로 옮겼다. 학생의 숫자는 Julia 난수로 학습한 학생의 것이다.
ENV["CH10_MODEL"] = "mace"
include("ch10_common2.jl")

const SETS = [("T1500", "104_test.npz"), ("T2500", "104_test_2500K.npz"), ("T1800", "104_transfer_1800K.npz"),
              ("strain", "104_transfer_strain.npz"), ("vacancy", "104_transfer_vacancy.npz")]

"동경 분포 함수 g(r): 0–rmax를 nb칸으로, 모든 순서쌍(i ≠ j와 다른 이미지)의 거리 히스토그램을 이상 기체의 껍질 부피로 나눈다"
function rdf(frames; rmax = 5.0, nb = 100)
    edges = range(0, rmax; length = nb + 1); h = zeros(nb); dr = step(edges)
    for f in frames
        for d in pair_distances(f.pos, f.H, rmax)
            k = floor(Int, d / dr) + 1
            1 <= k <= nb && (h[k] += 1)
        end
    end
    V = abs(det(frames[1].H)); n = natoms(frames[1])
    r = (edges[1:end-1] .+ edges[2:end]) ./ 2
    return collect(r), h ./ (length(frames) * n * (n / V) .* 4pi .* r .^ 2 .* dr)
end

"""Γ점 광학 진동수(THz): 8원자 셀에서 (0,0,0) fcc 부격자를 x로 ±0.01 Å 옮기고, 그 부격자 원자들의 평균 힘의 중심 차분으로
k = -ΔF/(2u)를 얻는다. 두 부격자의 환산 질량 μ = m_Si / 2, ν = √(k/μ) / 2π"""
function gamma_phonon(efun)
    s = si_cubic(A0_SI); u = 0.01
    sub = [i for i in 1:8 if all(isapprox.(mod.(frac(s)[:, i] .* 4, 2), 0; atol = 1e-8))]   # 1, 3, 5, 7번
    Fx = map((1, -1)) do sg
        R = copy(s.R); R[1, sub] .+= sg * u
        mean(efun(Cell(s.Z, R, s.h))[2][1, sub])
    end
    k = -(Fx[1] - Fx[2]) / (2u)
    return sqrt(k / (MSI / 2)) * THZ
end

function main()
    t0 = cputime()
    pool = load_frames("104_pool.npz")
    sets = [(name, load_frames(f)) for (name, f) in SETS]
    test = sets[1][2]

    # ---------------- Part A: 시간 상관된 프레임을 통한 누설
    traj = [f for f in pool if f.T == 1500 && f.step > 20]                  # 2 fs 간격의 연속 프레임 180개
    tr_idx = sort(randperm(Xoshiro(0), 120)[1:24])                          # 학습 프레임은 앞의 120개(21–140단계)에서
    near = [k for k in 1:120 if !(k in tr_idx) && minimum(abs.(tr_idx .- k)) <= 2]
    block = 141:180                                                          # 마지막 80 fs, 학습에 쓰지 않는다
    e0 = mean(f.E / 64 for f in traj)
    m, ps, st = new_student(0, e0)
    ps = train_frames!(m, ps, st, traj[tr_idx], 120; seed = 0)
    eA = Dict("near" => errors(m, ps, st, traj[near]), "block" => errors(m, ps, st, traj[block]), "indep" => errors(m, ps, st, test))
    println("Part A, one student trained on 24 frames of a single 1500 K oracle trajectory:")
    @printf("  held-out frames within 4 fs of a training frame (%d): F-RMSE %6.1f meV/A, E %5.2f meV/atom\n", length(near), eA["near"][2], eA["near"][1])
    @printf("  later time block of the same trajectory (40):        F-RMSE %6.1f meV/A, E %5.2f meV/atom\n", eA["block"][2], eA["block"][1])
    @printf("  independent 1500 K trajectory (21):                  F-RMSE %6.1f meV/A, E %5.2f meV/atom\n", eA["indep"][2], eA["indep"][1])

    # ---------------- Part B: 성적표
    orc = oracle("float32")
    calcs = Dict{String,Any}("oracle" => nothing)
    for arm in ("qbc", "rand", "aimd")
        calcs[arm] = load_model(joinpath(JOUT, "104_al_$(arm).jls"))
    end
    efun_of(name) = name == "oracle" ? (s -> calc_efs(orc, s.Z, s.R, s.h)[1:2]) :
                    (s -> student_ef(calcs[name]..., s.Z, s.R, s.h))
    ref_frames = [f for f in load_frames("104_pool_hot.npz") if f.T == 2500 && f.step > 50]

    Z, R64, H64 = si64()
    tt = Dict{String,Float64}()
    for name in ("oracle", "qbc")
        ef = efun_of(name); R = R64 .+ 0.05 .* randn(Xoshiro(1), size(R64)); ef(Cell(Z, R, H64))
        t = cputime()
        for _ in 1:10
            R .+= 0.001; ef(Cell(Z, R, H64))
        end
        tt[name] = (cputime() - t) / 10
    end
    @printf("CPU time per 64-atom force call: oracle (MACE-MP-0 small, float32) %.3f s, student SchNet %.1f ms, ratio %.0fx\n",
            tt["oracle"], 1000 * tt["qbc"], tt["oracle"] / tt["qbc"])

    card = Dict{String,Any}()
    for name in ("oracle", "qbc", "rand", "aimd")
        ef = efun_of(name); row = Dict{String,Any}()
        if name != "oracle"
            for (k, s) in sets
                row[k] = errors(calcs[name]..., s)[2]
            end
        end
        V, E = Float64[], Float64[]
        for a in range(5.38, 5.56; length = 7)
            c = si_cubic(a; rep = 2); push!(V, volume(c)); push!(E, ef(c)[1])
        end
        v0, _, B = sjeos(V, E)
        row["a0"] = cbrt(v0 / 8); row["B"] = B * GPA
        row["nu_gamma"] = gamma_phonon(ef)
        perf = Cell(Z, R64, H64); Ep = ef(perf)[1]
        vac = fire(ef, delete_atom(perf, 1); fmax = 0.02, maxsteps = 300)
        row["Evac"] = vac.E - 63 / 64 * Ep
        if name == "oracle"
            r, g = rdf(ref_frames)
        else
            fr, fail = langevin([calcs[name]], R64, H64, Z, 2500, 500; seed = 77, every = 2)
            row["md_fail"] = fail
            r, g = length(fr) > 60 ? rdf([f for f in fr if f.step > 100]) : (nothing, nothing)
        end
        row["r"] = r; row["rdf"] = g
        card[name] = row
        head = name == "oracle" ? "" : join([@sprintf("F[%s] %5.0f", k, row[k]) for (k, _) in sets], " | ") * " meV/A | "
        tail = name == "oracle" ? "" : ", MD fail step $(something(row["md_fail"], "None"))"
        @printf("%-6s: %sa0 %.4f A, B %5.1f GPa, nu_Gamma %5.2f THz, E_vac %.2f eV%s\n", name, head, row["a0"], row["B"], row["nu_gamma"], row["Evac"], tail)
    end
    r, gref = card["oracle"]["r"], card["oracle"]["rdf"]; i1 = argmax(gref)
    for name in ("qbc", "rand", "aimd")
        g = card[name]["rdf"]; g === nothing && continue
        @printf("  RDF %s: first-peak height %.2f at %.3f A (oracle %.2f at %.3f A); integrated |g - g_ref| dr = %.3f A\n",
                name, maximum(g), r[argmax(g)], gref[i1], r[i1], sum(abs.(g .- gref)) * (r[2] - r[1]))
    end
    card["partA"] = eA
    writejson(joinpath(JOUT, "104_val.json"), card)
    @printf("CPU %.0f s\n", cputime() - t0)
end
main()
