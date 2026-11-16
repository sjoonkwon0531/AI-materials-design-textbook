# lab_10_6.jl, Lab 10.6: 64원자 실리콘에서 라벨 16개를 고르는 네 가지 방법 (labs/104_data.py)
# 자료는 파이썬 labs/104_pool.py가 만든 오라클(MACE-MP-0 small) 자료를 그대로 읽는다(numpy, PythonCall):
#   후보 600프레임(300, 900, 1500 K Langevin MD, 2 fs), 흔들기 구조 16개, 시험 1500 K 21프레임, 1800 K 7프레임, 변형 세포 6개.
# (1) 힘의 시간 자기상관(Julia). (2) SOAP 기술자는 DScribe(PythonCall)로 만들고, 표준화, 최원점 선택(FPS), 최근접 거리, PCA는 Julia.
# (3) 방법마다 학생 SchNet을 120 에포크 학습(Adam, lr 2e-3 코사인, 손실 = 원자당 에너지 오차² + 10 x 힘 오차²), 시드 3개.
# PY_SELECTION = true면 block, random의 번호와 FPS의 시작점을 파이썬 시드 0–2가 뽑은 값으로 쓴다(아래 상수, 0부터 센 번호).
# 그러면 고른 구조와 SOAP 덮개 거리는 파이썬과 같고, 학생의 초기화와 배치 순서만 Julia 난수가 된다.
include("ch10_common.jl")

const PY_SELECTION = true
const PY_BLOCK_START = [163, 113, 162]                     # i1500[st:st+16]의 st
const PY_RANDOM = [[299, 44, 24, 158, 543, 300, 436, 9, 362, 372, 103, 385, 181, 580, 379, 482],
                   [560, 386, 251, 245, 153, 84, 20, 557, 147, 485, 494, 516, 162, 299, 442, 184],
                   [433, 198, 153, 526, 266, 355, 334, 112, 591, 64, 175, 54, 33, 483, 243, 479]]
const PY_FPS_START = [53, 586, 560]

"DScribe SOAP(rcut 4.5, nmax 6, lmax 4, σ 0.5, 주기, average = inner): 프레임마다 기술자 하나, d x n"
function soap_matrix(frames)
    soap = pyimport("dscribe.descriptors").SOAP(; species = pylist(["Si"]), r_cut = 4.5, n_max = 6, l_max = 4, sigma = 0.5,
                                                periodic = true, average = "inner")
    X = jarr(Float64, soap.create(pylist([to_ase(f.Z, f.pos, f.H) for f in frames]); n_jobs = 1))
    return permutedims(X)
end

function main()
    t0 = cputime()
    pool = load_frames("104_pool.npz"); rat = load_frames("104_rattle.npz"); test = load_frames("104_test.npz")
    hot = load_frames("104_transfer_1800K.npz"); strain = load_frames("104_transfer_strain.npz")
    Ts = [Int(f.T) for f in pool]

    println("force autocorrelation <F(t).F(t+lag)>/<F.F> along oracle MD (dt = 2 fs)")
    acf = Dict{Int,Vector{Float64}}()
    for T in (300, 900, 1500)
        Fs = [f.F for f in pool if f.T == T][51:end]                  # 처음 100 fs를 버린다
        c = force_acf(Fs, 40); acf[T] = c
        lag01 = findfirst(<(0.1), c) - 1                              # lag는 0부터 센다
        @printf("  %5d K: lag 1 = %.3f, lag 5 = %.3f, lag 10 = %.3f; first below 0.1 at lag %d (%d fs)\n",
                T, c[2], c[6], c[11], lag01, 2lag01)
    end

    X = soap_matrix(pool); Xr = soap_matrix(rat); Xt = soap_matrix(test)
    mu = mean(X; dims = 2); sd = std(X; dims = 2, corrected = false) .+ 1e-12      # numpy std와 같은 모집단 표준편차
    Zp, Zr, Zt = (X .- mu) ./ sd, (Xr .- mu) ./ sd, (Xt .- mu) ./ sd
    println("SOAP: $(size(X, 1)) components per frame")

    N = 16; i1500 = findall(==(1500), Ts)
    strategies = Dict{Tuple{String,Int},Any}()
    for seed in 0:2
        rng = Xoshiro(seed)
        st = PY_SELECTION ? PY_BLOCK_START[seed+1] : rand(rng, 50:length(i1500)-N-1)
        strategies[("block", seed)] = i1500[st+1:st+N]                              # 연속한 1500 K 16프레임
        strategies[("random", seed)] = PY_SELECTION ? PY_RANDOM[seed+1] .+ 1 : randperm(rng, length(pool))[1:N]
        strategies[("fps", seed)] = fps(Zp, N, PY_SELECTION ? PY_FPS_START[seed+1] + 1 : rand(rng, 1:length(pool)))
        strategies[("rattle", seed)] = nothing                                       # 흔들기 구조 16개
    end

    e0 = mean(f.E / natoms(f) for f in pool)
    res = Dict{String,Vector{Any}}()
    for name in ("block", "random", "fps", "rattle"), seed in 0:2
        idx = strategies[(name, seed)]
        tr = idx === nothing ? rat : pool[idx]
        cover = median(nn_dist(idx === nothing ? Zr : Zp[:, idx], Zt))
        m, ps, st = new_student(seed, e0)
        ps = train_frames!(m, ps, st, tr, 120; seed)
        r = (test = errors(m, ps, st, test), hot = errors(m, ps, st, hot), strain = errors(m, ps, st, strain), cover = cover)
        push!(get!(res, name, Any[]), r)
        @printf("  %-6s seed %d: test-1500K F-RMSE %6.1f meV/A | 1800K %6.1f | strained %6.1f | E-RMSE %5.2f meV/atom | SOAP NN-dist to test %5.2f\n",
                name, seed, r.test[2], r.hot[2], r.strain[2], r.test[1], cover)
    end
    println("summary (mean +/- std over 3 seeds), force RMSE in meV/A:")
    for name in ("block", "random", "fps", "rattle")
        a = [[r.test[2], r.hot[2], r.strain[2], r.test[1], r.cover] for r in res[name]]
        A = reduce(hcat, a); mm = vec(mean(A; dims = 2)); ss = vec(std(A; dims = 2, corrected = false))
        @printf("  %-6s: 1500K %6.1f +/- %4.1f | 1800K %6.1f +/- %4.1f | strained %6.1f +/- %4.1f | E %5.2f meV/atom | cover %.2f\n",
                name, mm[1], ss[1], mm[2], ss[2], mm[3], ss[3], mm[4], mm[5])
    end
    f0 = strategies[("fps", 0)]
    println("FPS (seed 0) picks per temperature: ", Dict(T => count(==(T), Ts[f0]) for T in (300, 900, 1500)))
    Zc = Zp .- mean(Zp; dims = 2); sv = svdvals(Zc)
    @printf("PCA: first two components explain %.1f %% of SOAP variance\n", 100 * sum(abs2, sv[1:2]) / sum(abs2, sv))
    @printf("CPU %.0f s\n", cputime() - t0)
end

main()
