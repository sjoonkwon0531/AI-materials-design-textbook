# lab_10_9.jl: Lab 10.9: MACE-MP-0 small을 다른 기준 계산(GFN2-xTB)에 맞추기 (labs/105_ft.py)
# 실행: julia lab_10_9.jl      (자료 data/105_xtb_{train,test}.npz는 파이썬 labs/105_ft_data.py가 tblite로 만든 것을 그대로 읽는다)
# MACE는 Julia 구현이 없으므로 모델, 텐서, 역전파는 PythonCall로 부르는 torch다. Julia로 옮긴 것은 그 바깥의 논리 전부다:
# 원소 기준 에너지 E₀의 재조정(숫자 하나), 학습 가능한 매개변수의 선택(readout만 또는 전부), 배치 고르기, 손실
# 100 (ΔE/N)² + |ΔF|², PyTorch CosineAnnealingLR과 같은 학습률 일정, 무작위 재초기화(scratch), 시험 오차, 'sj' 상태방정식,
# 다른 다섯 물질에서의 망각 지표, 그리고 비교용 SchNet(ch10_common.jl의 학생, Julia로 학습).
# 난수: 배치 순서와 망각 탐침의 흔들기는 Julia 난수다. scratch의 가중치는 torch.Generator(시드 0)에서 뽑으므로 파이썬과 같다.
ENV["CH10_MODEL"] = "mace"
include("ch10_common2.jl")

const torch = pyimport("torch")
const mdata = pyimport("mace.data")
const tgeo = pyimport("mace.tools.torch_geometric")
const ttools = pyimport("mace.tools.torch_tools")
const pycopy = pyimport("copy")
const calc = oracle("float32")
const base = calc.models[0]
const keyspec = mdata.KeySpecification(; info_keys = calc.info_keys, arrays_keys = calc.arrays_keys)
const SI = calc.z_table.z_to_index(14)                     # Si의 원소 번호(모델 안의 순서)

# ---------------------------------------------------------------- MACE 입력과 출력
"Frame → MACE의 AtomicData (float32)"
function to_data(f::Frame)
    a = to_ase(f.Z, f.pos, f.H)
    return pywith(ttools.default_dtype(torch.float32)) do _
        cfg = mdata.config_from_atoms(a; key_specification = keyspec, head_name = calc.head)
        mdata.AtomicData.from_config(cfg; z_table = calc.z_table, cutoff = calc.r_max, heads = calc.available_heads)
    end
end
"모델 출력 (에너지 텐서 (B), 힘 텐서 (N, 3)). training = true면 역전파할 수 있는 그래프가 남는다"
function mace_run(model, frames; training = false)
    b = tgeo.batch.Batch.from_data_list(pylist([to_data(f) for f in frames]))
    out = model(b.to_dict(); training = training, compute_force = true)
    return out["energy"], out["forces"]
end
tojl(x) = jarr(Float64, x.detach().numpy())
"참값 텐서: 에너지 (B), 힘 (N, 3), 원자 수 (B)"
function targets(frames)
    Er = torch.tensor(pylist([f.E for f in frames]); dtype = torch.float32)
    Fr = torch.tensor(np.array(permutedims(reduce(hcat, [f.F for f in frames]))); dtype = torch.float32)
    nat = torch.tensor(pylist([natoms(f) for f in frames]); dtype = torch.float32)
    return Er, Fr, nat
end

"시험 오차: 원자당 에너지 RMSE (meV/atom), 힘 성분 RMSE (meV/Å)"
function mace_errors(model, frames)
    E, F = mace_run(model, frames)
    dE = (tojl(E) .- [f.E for f in frames]) ./ [natoms(f) for f in frames]
    dF = tojl(F) .- permutedims(reduce(hcat, [f.F for f in frames]))
    return 1000 * sqrt(mean(abs2, dE)), 1000 * sqrt(mean(abs2, dF))
end

"""1단계: Si의 고립 원자 기준 에너지 E₀를 원자당 평균 에너지 차만큼 옮긴다. 원소가 Si 하나뿐이면 이것은 모든 구조의
에너지를 N d만큼 옮길 뿐 힘은 바꾸지 않는다(연습문제 10.5.2). 옮긴 양 d(eV)를 돌려준다."""
function refit_e0!(model, frames)
    E = tojl(first(mace_run(model, frames)))
    d = mean((f.E - e) / natoms(f) for (f, e) in zip(frames, E))
    ae = model.atomic_energies_fn.atomic_energies
    ae[SI] = ae[SI].add(d)                                 # 버퍼(학습 대상이 아님)를 제자리에서 바꾼다
    return d
end

"Si 다이아몬드 8원자 셀 a = 5.36–5.66 Å의 7점, 'sj' 상태방정식: (a₀, B [GPa])"
function mace_eos(model)
    V, E = Float64[], Float64[]
    for a in range(5.36, 5.66; length = 7)
        c = si_cubic(a); f = Frame(; pos = c.R, H = c.h, Z = c.Z)
        push!(V, volume(c)); push!(E, tojl(first(mace_run(model, [f])))[1])
    end
    v0, _, B = sjeos(V, E)
    return cbrt(v0), B * GPA
end

"""미세조정: params만 학습한다. 매 단계 bs개를 비복원으로 뽑고, 손실 wE ⟨(ΔE/N)²⟩ + wF ⟨ΔF²⟩, Adam,
학습률 lr → 0.05 lr의 코사인(PyTorch CosineAnnealingLR(T_max = steps, eta_min = 0.05 lr)과 같은 값)."""
function finetune!(model, frames, params; steps = 40, lr = 1e-3, bs = 4, seed = 0, wE = 100.0, wF = 1.0)
    for p in model.parameters()
        p.requires_grad_(false)
    end
    for p in params
        p.requires_grad_(true)
    end
    opt = torch.optim.Adam(pylist(params); lr = lr)
    rng = Xoshiro(seed)
    for s in 0:steps-1
        fr = frames[randperm(rng, length(frames))[1:min(bs, length(frames))]]
        E, F = mace_run(model, fr; training = true)
        Er, Fr, nat = targets(fr)
        loss = E.sub(Er).div(nat).pow(2).mean().mul(wE).add(F.sub(Fr).pow(2).mean().mul(wF))
        for g in opt.param_groups
            g["lr"] = cosine_lr(lr, s, steps; eta_min = 0.05lr)
        end
        opt.zero_grad(); loss.backward(); opt.step()
    end
    for p in model.parameters()
        p.requires_grad_(true)
    end
    return model
end

"scratch: 모든 가중치를 표준정규분포에서 다시 뽑는다(e3nn 층의 기본 초기화와 같은 분포)"
function reinit_randn!(m)
    g = torch.Generator().manual_seed(0)
    pywith(torch.no_grad()) do _
        for p in m.parameters()
            p.copy_(torch.randn(p.shape; generator = g))
        end
    end
    return m
end

# ---------------------------------------------------------------- 망각 탐침: 미세조정에 없는 다섯 물질, 흔든 세포 각 3개
function probe_frames()
    build = pyimport("ase.build"); rng = Xoshiro(3)
    specs = [("C", build.bulk("C", "diamond"; a = 3.57, cubic = true)), ("SiC", build.bulk("SiC", "zincblende"; a = 4.38, cubic = true)),
             ("Ge", build.bulk("Ge", "diamond"; a = 5.76, cubic = true)), ("Al", build.bulk("Al", "fcc"; a = 4.04, cubic = true).repeat((2, 1, 1))),
             ("NaCl", build.bulk("NaCl", "rocksalt"; a = 5.70, cubic = true))]
    out = Tuple{String,Frame}[]
    for (name, a) in specs
        Z, R, H = from_ase(a)
        for _ in 1:3
            push!(out, (name, Frame(; pos = R .+ 0.06 .* randn(rng, size(R)), H, Z)))
        end
    end
    return out
end
const PROBES = probe_frames()
const F_ORIG = [tojl(last(mace_run(base, [f]))) for (_, f) in PROBES]
"원래 MACE-MP-0의 힘으로부터 벗어난 정도: 물질마다 √(탐침 셋에 대한 ⟨ΔF²⟩의 평균), meV/Å"
function drift(model)
    out = Dict{String,Vector{Float64}}()
    for ((name, f), F0) in zip(PROBES, F_ORIG)
        push!(get!(out, name, Float64[]), mean(abs2, tojl(last(mace_run(model, [f]))) .- F0))
    end
    return Dict(k => 1000 * sqrt(mean(v)) for (k, v) in out)
end
drift_str(dr) = join([@sprintf("%.0f", dr[k]) for k in ("C", "SiC", "Ge", "Al", "NaCl")], "/")

function main()
    t0 = cputime()
    train_all = load_frames("105_xtb_train.npz"); test = load_frames("105_xtb_test.npz")
    ref = readjson(joinpath(DATA, "105_xtb_meta.json"))["ref"]
    groups = Dict{String,Int}()
    for kv in base.named_parameters()
        k = split(pyconvert(String, kv[0]), ".")[1]; groups[k] = get(groups, k, 0) + pyconvert(Int, kv[1].numel())
    end
    @printf("MACE-MP-0 small: %d parameters %s; r_max %.1f A, %d interaction layers, E0(Si) %.4f eV\n", sum(values(groups)), string(groups),
            pyconvert(Float64, base.r_max), pyconvert(Int, base.num_interactions), pyconvert(Float64, base.atomic_energies_fn.atomic_energies[SI]))
    @printf("MPtrj (CHGNet paper numbers): %.1f force vectors per configuration on average\n", 49295660 / 1580395)
    @printf("GFN2-xTB reference EOS: a0 %.4f A, B %.1f GPa\n", ref["a0"], ref["B_GPa"])
    z0 = pycopy.deepcopy(base)
    e_raw, f_raw = mace_errors(z0, test); a0_0, B_0 = mace_eos(z0)
    @printf("MACE-MP-0 small as is   : test E %7.1f meV/atom, F %5.0f meV/A | a0 %.4f A, B %5.1f GPa\n", e_raw, f_raw, a0_0, B_0)

    report(N, method, m, d, tm) = begin
        e, f = mace_errors(m, test); a0, B = mace_eos(m); dr = drift(m)
        @printf("N=%2d %-9s: test E %6.2f meV/atom, F %5.0f meV/A | a0 %.4f, B %6.1f GPa | drift on C/SiC/Ge/Al/NaCl %s meV/A | dE0 %+.3f eV | %.0f s\n",
                N, method, e, f, a0, B, drift_str(dr), d, cputime() - tm)
        Dict("N" => N, "method" => method, "E" => e, "F" => f, "a0" => a0, "B" => B, "drift" => dr, "dE0" => d)
    end
    results = Any[]
    for N in (4, 16, 48), method in ("zero-shot", "readout", "full", "scratch")
        tm = cputime(); tr = train_all[1:N]
        m = pycopy.deepcopy(base)
        method == "scratch" && reinit_randn!(m)
        d = refit_e0!(m, tr)
        if method == "readout"
            finetune!(m, tr, [kv[1] for kv in m.named_parameters() if startswith(pyconvert(String, kv[0]), "readouts")]; lr = 5e-3)
        elseif method in ("full", "scratch")
            finetune!(m, tr, collect(m.parameters()); lr = 1e-3)
        end
        push!(results, report(N, method, m, d, tm))
    end
    for method in ("full", "scratch")                     # N = 48에서 예산을 다섯 배(200단계)로
        tm = cputime(); m = pycopy.deepcopy(base)
        method == "scratch" && reinit_randn!(m)
        d = refit_e0!(m, train_all)
        finetune!(m, train_all, collect(m.parameters()); steps = 200, lr = 1e-3)
        push!(results, report(48, method * "-200", m, d, tm))
    end

    # 값싼 모델의 기준: SchNet을 같은 48개로 처음부터 300 에포크 (§10.4의 손실 가중치 E 1, F 10 그대로)
    e0 = mean(f.E / 8 for f in train_all)
    sm, sps, sst = new_student(0, e0)
    sps = train_frames!(sm, sps, sst, train_all, 300; seed = 0)
    es, fs = errors(sm, sps, sst, test)
    V, E = Float64[], Float64[]
    for a in range(5.36, 5.66; length = 7)
        c = si_cubic(a); push!(V, volume(c)); push!(E, first(student_ef(sm, sps, sst, c.Z, c.R, c.h)))
    end
    v0s, _, Bs = sjeos(V, E)
    @printf("SchNet from scratch, 48 labels, 300 epochs: test E %.2f meV/atom, F %.0f meV/A | a0 %.4f, B %.1f GPa\n", es, fs, cbrt(v0s), Bs * GPA)
    n_read = sum(pyconvert(Int, kv[1].numel()) for kv in base.named_parameters() if startswith(pyconvert(String, kv[0]), "readouts"))
    @printf("trainable parameters: readout %d, full %d\n", n_read, sum(values(groups)))
    writejson(joinpath(JOUT, "105_ft.json"), Dict("results" => results, "raw" => Dict("E" => e_raw, "F" => f_raw, "a0" => a0_0, "B" => B_0),
                                                   "schnet" => Dict("E" => es, "F" => fs, "a0" => cbrt(v0s), "B" => Bs * GPA)))
    @printf("CPU %.0f s\n", cputime() - t0)
end
main()
