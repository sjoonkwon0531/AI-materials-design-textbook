# lab_10_3.jl, Lab 10.3: 보존력 대 직접 힘, 같은 자료로 학습한 두 모델의 NVE (labs/102_nve.py)
# (A) EqNet 등뼈(rc = 3.0 Å, 층 2, 채널 32)에 두 머리: cons(F = -∂E/∂R, 자동미분)와 direct(별도의 등변 벡터 머리 F_i = W v_i).
#     8원자 Si 셀 400개, Adam 2,500단계, 배치 16, lr 3e-3, wE = 10, wF = 1, 시드 0.
# (B) 64원자(2 x 2 x 2) 셀의 속도 Verlet NVE, 600 K Maxwell–Boltzmann 시작(총 운동량 0, 온도를 정확히 맞춤), 5 ps,
#     Δt = 1.0과 0.5 fs. E_model + K를 10걸음마다, 오라클(MACE-MP-0 small, PythonCall)로 다시 잰 E_oracle + K를 250 fs마다 기록.
# (C) 완전 결정 8원자 셀의 Hessian: 이중 자동미분(Zygote.hessian) 대 힘의 유한차분, 그리고 오라클의 유한차분 Hessian과 진동수 비교.
include("ch10_common.jl")

function md(m, ps, st, R0, H, dt, t_ps, calc; seed = 0, T0 = 600.0, every_fs = 250.0)
    n = size(R0, 2); mass = MSI * AMU; Z = fill(14, n)
    v0 = mb_velocities(Xoshiro(seed), n, T0, mass; exact = true)
    lists = [neighbor_list(R0, H, m.rc + 0.8)]                         # skin 0.8 Å의 고정 이웃 목록(파이썬 판과 같다)
    force(R) = (r = predict(m, ps, st, [Z], [R], [H]; lists); (r[1][1], r[2]))
    nsteps = round(Int, t_ps * 1000 / dt); stride = round(Int, every_fs / dt)
    t, Etot, Tk, to, Eo = Float64[], Float64[], Float64[], Float64[], Float64[]
    function rec(s, R, v, E)
        K = kinetic(v, mass)
        if s % 10 == 0
            push!(t, s * dt / 1000); push!(Etot, E + K); push!(Tk, temperature(v, mass))
        end
        if s % stride == 0
            push!(to, s * dt / 1000); push!(Eo, calc_energy(calc, Z, R, H) + K)
        end
    end
    c0 = cputime()
    R, _, _ = velocity_verlet(force, R0, v0, mass, dt, nsteps; record = rec)
    cpu = cputime() - c0
    dE = (Etot .- Etot[1]) ./ n .* 1e3; dEo = (Eo .- Eo[1]) ./ n .* 1e3   # meV/atom
    return (t = t, E = dE, T = Tk, to = to, Eo = dEo, cpu = cpu,
            disp = maximum(abs.(R .- R0)), com = norm(mean(R .- R0; dims = 2)))
end

function main()
    tr, te = load_si8(); e0 = mean(tr.E) / 8
    models = Dict{Symbol,Any}()
    for head in (:energy, :direct)
        m, ps, st = setup_model(Student(:eq, 3.0; head, e0); seed = 0)
        ps, cpu = train_si8!(m, ps, st, tr, collect(1:400); steps = 2500, bs = 16, wE = 10.0, wF = 1.0, seed = 0)
        ev = evaluate_si8(m, ps, st, te)
        net = [norm(sum(ev.F[:, :, b]; dims = 2)) for b in axes(ev.F, 3)]     # |Σ_i F_i| (보존력이면 반올림 수준)
        @printf("    %-6s: |sum_i F_i| over test cells: mean %.2e meV/A, max %.2e meV/A\n", head, 1e3mean(net), 1e3maximum(net))
        @printf("(A) %-6s: params %d  test E MAE %.1f meV/atom  F MAE %.0f meV/A  train cpu %.0f s\n",
                head, nparams(m), ev.E_mae, ev.F_mae, cpu)
        save_model(joinpath(JOUT, "102_nve_$(head).jls"), m, ps, st)
        models[head] = (m, ps, st)
    end
    calc = oracle("float64")
    _, R64, H64 = bulk_cell("Si", "diamond", 5.43; n = 2)
    res = Dict{String,Any}()
    for head in (:energy, :direct), dt in (1.0, 0.5)
        m, ps, st = models[head]
        r = md(m, ps, st, R64, H64, dt, 5.0, calc)
        sl = slope(r.t, r.E); slo = slope(r.to, r.Eo); fl = detrended_std(r.t, r.E)
        Tm = mean(r.T[length(r.T)÷2+1:end])
        @printf("(B) %-6s dt=%.1f fs: t_end %.2f ps | drift of E_model+K %+8.3f meV/atom/ps, fluct %.3f meV/atom | drift of E_oracle+K %+8.3f meV/atom/ps | <T> 2nd half %5.0f K | max disp %.2f A, centre-of-mass shift %.2f A | cpu %.0f s\n",
                head, dt, r.t[end], sl, fl, slo, Tm, r.disp, r.com, r.cpu)
        res["$(head)_dt$dt"] = Dict("t" => r.t, "E" => r.E, "to" => r.to, "Eo" => r.Eo, "drift" => sl, "fluct" => fl)
    end
    # (C) 2계 미분
    Z8, R8, H8 = bulk_cell("Si", "diamond", 5.43)
    m, ps, st = models[:energy]
    Had = hessian_ad(m, ps, st, Z8, R8, H8)
    lists = [neighbor_list(R8, H8, m.rc + 0.3)]
    Hfd = hessian_fd(R -> predict(m, ps, st, [Z8], [R], [H8]; lists)[2], R8)
    @printf("(C) student Hessian: max |AD - FD| = %.1e eV/A^2 (max |H| = %.2f); asymmetry of AD Hessian %.1e\n",
            maximum(abs.(Had .- Hfd)), maximum(abs.(Had)), maximum(abs.(Had .- Had')))
    c0 = cputime(); Hor = hessian_fd(R -> calc_efs(calc, Z8, R, H8)[2], R8); cpu_or = cputime() - c0
    fs, fo = freqs_thz((Had .+ Had') ./ 2), freqs_thz(Hor)
    @printf("    oracle Hessian by FD: %d force calls, cpu %.0f s\n", 2 * 24, cpu_or)
    println("    8-atom-cell modes (THz), highest four: student ", round.(fs[end-3:end]; digits = 2), "  oracle ", round.(fo[end-3:end]; digits = 2))
    println("    three acoustic (should be 0): student ", round.(fs[1:3]; digits = 3), "  oracle ", round.(fo[1:3]; digits = 3))
    @printf("    mean |student - oracle| over the 21 non-acoustic modes: %.2f THz\n", mean(abs.(fs[4:end] .- fo[4:end])))
    res["freqs"] = Dict("student" => fs, "oracle" => fo)
    writejson(joinpath(JOUT, "102_nve.json"), res)
end

main()
