# lab_10_4.jl, Lab 10.4: 짧은 절단 반경에서 l = 1 채널이 하는 일 (labs/102_angular.py)
# 같은 자료(8원자 Si 셀 400개), 같은 예산에서 InvNet(l = 0)과 EqNet(l ≤ 1)을 rc = 3.0, 4.2 Å로 학습하고 시험 셀 200개에서 잰다.
# 두 모델 모두 층 2, 채널 32, Bessel 기저 8. Adam 2,500단계, 배치 16, lr 3e-3 코사인, wE = 10, wF = 1, 시드 0.
include("ch10_common.jl")

function main()
    tr, te = load_si8(); e0 = mean(tr.E) / 8
    for rc in (3.0, 4.2)                                             # 이웃 수: 첫째 껍질(4개)만, 둘째 껍질(12개)까지
        nnb = mean(length(first(neighbor_list(te.R[:, :, b], te.H[:, :, b], rc))) for b in axes(te.R, 3)) / 8
        @printf("rc = %.1f A: mean neighbours per atom = %.2f\n", rc, nnb)
    end
    @printf("zero-force baseline on test: F MAE = %.0f meV/A\n", 1e3 * mean(abs.(te.F)))
    res = Dict{String,Any}()
    for rc in (3.0, 4.2), kind in (:inv, :eq)
        m, ps, st = setup_model(Student(kind, rc; e0); seed = 0)
        ps, cpu = train_si8!(m, ps, st, tr, collect(1:400); steps = 2500, bs = 16, wE = 10.0, wF = 1.0, seed = 0)
        ev = evaluate_si8(m, ps, st, te)
        res["$(kind)_rc$(rc)_N400"] = Dict("E_mae" => ev.E_mae, "F_mae" => ev.F_mae, "cpu" => cpu, "params" => nparams(m))
        @printf("%-3s rc=%.1f N=400 params=%6d | E MAE %6.1f meV/atom  F MAE %5.0f meV/A | cpu %5.1f s\n",
                kind, rc, nparams(m), ev.E_mae, ev.F_mae, cpu)
    end
    writejson(joinpath(JOUT, "102_angular.json"), res)
end

main()
