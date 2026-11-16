# lab_10_7.jl, Lab 10.7: 위원회 능동 학습 대 무작위 선택 대 오라클 MD (labs/104_al.py, 104_al_summary.py)
#   julia lab_10_7.jl [seed]     한 시드의 세 방식 (seed 0이면 마지막 학생을 Lab 10.8을 위해 models/104_al_<arm>.jls로 저장)
#   julia lab_10_7.jl summary    시드 0과 1의 결과를 합친 표
# 오라클(MACE-MP-0 small, float32)은 PythonCall로 부른다. 그 밖의 모든 것은 Julia다: 학생 SchNet 위원회, BAOAB Langevin 탐색,
# 식 (10.4.1)의 ε_F, DP-GEN 신뢰 구간 [0.05, 1.5) eV/Å, 같은 줄기에서 5걸음 이상 떨어지게 고르는 탐욕 선택, 이어 학습, 평가.
# 세 방식 모두 흔들기 구조 4개에서 시작해 다섯 라운드(탐색 온도 900, 1500, 2000, 2500, 2500 K) 동안 라운드마다 라벨 6개를 더한다.
include("ch10_common.jl")

const TEMPS = [900, 1500, 2000, 2500, 2500]
const T_TARGET, MD_STEPS, K, LO, HI = 2500, 500, 6, 0.05, 1.5
const EPOCHS0, EPOCHS = 100, 40

"2500 K MD 두 줄기(각 500걸음)가 무너지지 않고 버틴 걸음 수"
function stability(member, x0, H, Z)
    return [something(last(langevin([member], x0, H, Z, T_TARGET, MD_STEPS; seed = 900 + s, every = 10^6)), MD_STEPS) for s in 0:1]
end

"학생 MD 두 줄기(각 150걸음)로 후보를 모은다. committee면 매 걸음 위원회의 ε_F를 잰다. 무너진 구조도 후보에 넣는다."
function explore(models, x0, H, Z, T, rnd, seed; committee)
    cand = Frame[]; fails = Any[]
    for tr in 0:1
        fr, fail = langevin(committee ? models : models[1:1], x0, H, Z, T, 150; seed = 1000rnd + tr + 7seed, every = 1,
                            committee, keep_fail = true)
        foreach(f -> (f.traj = tr), fr)
        append!(cand, fr); push!(fails, fail)
    end
    return cand, fails
end

"신뢰 구간 안에서 ε_F가 큰 순서로, 같은 줄기 안에서는 5걸음 이상 떨어진 프레임 K개 (구간이 비면 전체에서)"
function pick_qbc(cand)
    dev = [c.dev for c in cand]
    ok = findall(d -> LO <= d < HI, dev)
    order = isempty(ok) ? sortperm(-dev) : ok[sortperm(-dev[ok])]
    pick = Int[]
    for i in order
        if all(cand[i].traj != cand[j].traj || abs(cand[i].step - cand[j].step) >= 5 for j in pick)
            push!(pick, i)
        end
        length(pick) == K && break
    end
    return pick, dev, ok
end

function run_seed(SEED)
    t0 = cputime()
    orc = oracle("float32")
    pool = vcat(load_frames("104_pool.npz"), load_frames("104_pool_hot.npz"))
    rat = load_frames("104_rattle.npz"); test = load_frames("104_test_2500K.npz"); test1500 = load_frames("104_test.npz")
    Z, x0, H = si64()
    e0 = mean(f.E / 64 for f in rat)
    rng_sel = Xoshiro(5 + SEED)
    arms = Dict{String,Any}()
    for arm in ("qbc", "rand", "aimd")
        data = [copy(f) for f in rat[1:4]]
        models = Any[new_student(s + 100SEED, e0) for s in (arm == "qbc" ? (0:2) : (0:0))]
        for (k, (m, ps, st)) in enumerate(models)
            models[k] = (m, train_frames!(m, ps, st, data, EPOCHS0; seed = k - 1 + 100SEED), st)
        end
        hist = Any[]; oracle_calls = length(data); md_T = Set{Int}()
        for rnd in 0:length(TEMPS)
            err = errors(models[1]..., test); err15 = errors(models[1]..., test1500)
            push!(hist, Dict("round" => rnd, "labels" => length(data), "F_rmse" => err[2], "E_rmse" => err[1], "F_rmse_1500" => err15[2],
                             "stable_steps" => stability(models[1], x0, H, Z), "oracle_calls" => oracle_calls + 200 * length(md_T)))
            rnd == length(TEMPS) && break
            T = TEMPS[rnd+1]
            if arm == "aimd"                                   # 오라클 자신의 MD 프레임(탐색 온도 이하)에서 무작위로
                union!(md_T, [t for t in (300, 900, 1500, 2000, 2500) if t <= T])
                idx = [i for (i, f) in enumerate(pool) if Int(f.T) in md_T && f.step > 20]
                new = [copy(pool[i]) for i in idx[randperm(rng_sel, length(idx))[1:K]]]
                info = "oracle-MD frames at T in $(sort(collect(md_T)))"
            else
                cand, fails = explore(models, x0, H, Z, T, rnd, SEED; committee = arm == "qbc")
                if arm == "qbc"
                    pick, dev, ok = pick_qbc(cand)
                    info = @sprintf("explore MD failed at steps %s; %d candidates, dev median %.3f max %.3f eV/A, %d in window, picked dev %.3f-%.3f",
                                    string(fails), length(cand), median(dev), maximum(dev), length(ok), minimum(dev[pick]), maximum(dev[pick]))
                else
                    pick = randperm(rng_sel, length(cand))[1:min(K, length(cand))]
                    info = "explore MD failed at steps $(fails); $(length(cand)) candidates, random pick"
                end
                new = label(orc, cand[pick])                    # 비싼 단계는 이것뿐이다
                oracle_calls += length(new)
            end
            append!(data, new)
            for (k, (m, ps, st)) in enumerate(models)          # 이전 매개변수에서 이어 학습
                models[k] = (m, train_frames!(m, ps, st, data, EPOCHS; seed = 10rnd + k - 1 + 100SEED), st)
            end
            println("[$arm] round $(rnd + 1) T=$T K: $info; labels now $(length(data))")
        end
        arms[arm] = hist
        SEED == 0 && save_model(joinpath(JOUT, "104_al_$(arm).jls"), models[1]...)
        println("[$arm] ", join([@sprintf("n=%2d: F %5.0f (%4.0f) meV/A, stable %s", h["labels"], h["F_rmse"], h["F_rmse_1500"], string(h["stable_steps"])) for h in hist], " | "))
    end
    println("summary at equal labels (force RMSE on 2500 K test, meV/A; MD steps survived at 2500 K out of 500, two runs):")
    for r in 1:length(TEMPS)+1
        println(@sprintf("  labels %2d -> ", arms["qbc"][r]["labels"]),
                join([@sprintf("%s: %5.0f, %s", a, arms[a][r]["F_rmse"], string(arms[a][r]["stable_steps"])) for a in ("qbc", "rand", "aimd")], " | "))
    end
    println("oracle calls spent by the end: ", join(["$a $(arms[a][end]["oracle_calls"])" for a in ("qbc", "rand", "aimd")], ", "))
    writejson(joinpath(JOUT, SEED == 0 ? "104_al.json" : "104_al_s$(SEED).json"), arms)
    @printf("CPU %.0f s\n", cputime() - t0)
end

"두 시드의 요약 (labs/104_al_summary.py). FROM_PYTHON = true면 파이썬의 data/104_al.json, 104_al_s1.json을 읽는다."
function summary(; FROM_PYTHON = true)
    dir = FROM_PYTHON ? DATA : JOUT
    A = [readjson(joinpath(dir, "104_al.json")), readjson(joinpath(dir, "104_al_s1.json"))]
    println("labels | force RMSE 2500 K, mean of 2 seeds (meV/A) | stable MD steps at 2500 K, mean of 4 runs (min-max)")
    for r in 1:6
        row = map(("qbc", "rand", "aimd")) do arm
            F = mean(a[arm][r]["F_rmse"] for a in A); st = reduce(vcat, [Int.(a[arm][r]["stable_steps"]) for a in A])
            @sprintf("%s %4.0f, %3.0f (%d-%d)", arm, F, mean(st), minimum(st), maximum(st))
        end
        println(@sprintf("  %2d  | ", A[1]["qbc"][r]["labels"]), join(row, " | "))
    end
    for arm in ("qbc", "rand", "aimd")
        st = reduce(vcat, [Int.(a[arm][r]["stable_steps"]) for r in 4:6 for a in A])       # 라벨 22–34개인 세 라운드
        @printf("%s: rounds with 22-34 labels, %d of %d MD runs reached 500 steps; mean %.0f steps\n", arm, count(==(500), st), length(st), mean(st))
    end
end

isempty(ARGS) ? run_seed(0) : ARGS[1] == "summary" ? summary() : run_seed(parse(Int, ARGS[1]))
