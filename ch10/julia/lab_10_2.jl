# lab_10_2.jl, Lab 10.2: 라벨마다 무엇을 가르치는가, 그리고 응력을 자동미분으로 (labs/102_stress.py)
# (A) 자동미분 응력 대 중심 유한차분: MACE-MP-0 small(내장 응력, PythonCall)과 Julia EqNet(Derivation Box 10.2의 대칭 변형).
# (B) 같은 셀 수에서 손실 조합만 바꾸어 학습: E, E+F, F, 잡음 넣은 E+F, E+F+S. 학생은 EqNet(rc = 3.0 Å, 층 2, 채널 32),
#     Adam 1,500단계, 배치 8, lr 3e-3 코사인, wE = 10, wF = 1, wS = 100 (응력 단위 eV/Å³). 자료: data/102_si8.npz.
include("ch10_common.jl")

function part_a(tr, te)
    println("(A) stress: autodiff vs central finite differences (h = 1e-4), test cell 0")
    Z = fill(14, 8); R = te.R[:, :, 1]; H = te.H[:, :, 1]
    calc = oracle("float64")
    _, _, s_ad = calc_efs(calc, Z, R, H; forces = false, stress = true)
    s_fd = fd_stress((Rx, Hx) -> calc_energy(calc, Z, Rx, Hx), R, H)
    @printf("  MACE-MP-0 : sigma_xx %8.4f GPa (AD) %8.4f (FD); max |AD-FD| over 9 components = %.1e GPa\n",
            s_ad[1, 1] * GPA, s_fd[1, 1] * GPA, maximum(abs.(s_ad .- s_fd)) * GPA)
    m, ps, st = setup_model(Student(:eq, 3.0; e0 = mean(tr.E) / 8); seed = 0)       # 학습하지 않은 EqNet
    lists = [neighbor_list(R, H, 3.0 + 0.5)]                                         # 변형해도 바꾸지 않는 이웃 목록(skin 0.5 Å)
    e_student(Rx, Hx) = predict(m, ps, st, [Z], [Rx], [Hx]; lists)[1][1]
    _, _, S = predict(m, ps, st, [Z], [R], [H]; stress = true, lists)
    s_ad = S[:, :, 1]; s_fd = fd_stress(e_student, R, H)
    @printf("  EqNet     : max |AD-FD| = %.1e GPa  (symmetric: max|S-S^T| = %.1e GPa)\n",
            maximum(abs.(s_ad .- s_fd)) * GPA, maximum(abs.(s_ad .- s_ad')) * GPA)
    @printf("  oracle stress over test cells: mean |sigma_ab| = %.2f GPa\n", mean(abs.(te.S)) * GPA)
end

function part_b(tr, te)
    println("(B) what the labels buy (test: 200 cells; E in meV/atom, F in meV/A)")
    e0 = mean(tr.E) / 8; res = Dict{String,Any}()
    function run(tag, N; kw...)
        m, ps, st = setup_model(Student(:eq, 3.0; e0); seed = 0)
        ps, cpu = train_si8!(m, ps, st, tr, collect(1:N); steps = 1500, bs = 8, seed = 0, kw...)
        ev = evaluate_si8(m, ps, st, te; stress = true)
        d = (ev.E .- te.E) ./ 8                                       # 힘만으로는 정할 수 없는 상수를 따로 본다
        off = 1e3 * mean(d); shifted = 1e3 * mean(abs.(d .- mean(d)))
        res["$(tag)_N$N"] = Dict("E_mae" => ev.E_mae, "F_mae" => ev.F_mae, "S_mae" => ev.S_mae, "E_offset" => off,
                                 "E_mae_shifted" => shifted, "cpu" => cpu)
        @printf("  %-22s N=%3d | E MAE %7.1f (offset %+7.1f, after shift %5.1f) | F MAE %5.0f | S MAE %5.2f GPa | cpu %4.0f s\n",
                tag, N, ev.E_mae, off, shifted, ev.F_mae, ev.S_mae, cpu)
    end
    for N in (10, 40, 160)
        run("E only", N; wE = 10.0, wF = 0.0)
        run("E+F", N; wE = 10.0, wF = 1.0)
    end
    run("F only", 160; wE = 0.0, wF = 1.0)
    run("E+F, F noise 0.2 eV/A", 160; wE = 10.0, wF = 1.0, noiseF = 0.2)
    run("E+F+S", 40; wE = 10.0, wF = 1.0, wS = 100.0)
    @printf("  labels per 8-atom cell: energy 1, forces %d, stress 6 -> E+F carries %dx the scalars of E only\n", 3 * 8, 1 + 3 * 8)
    writejson(joinpath(JOUT, "102_stress.json"), res)
end

tr, te = load_si8()
part_a(tr, te)
part_b(tr, te)
