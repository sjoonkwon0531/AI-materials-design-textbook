# lab_9_18.jl — Lab 9.18: 세 해법(PINN, FNO, 고전 솔버)의 정확도, 시간, 손익분기 (labs/95_compare.py, 95_pinn_ns.py)
# 기준: 시험 초기조건 8개의 128² 의사 스펙트럴 해(dt = 2.5e-3)를 64²로 자른 것. 모든 해법을 64² 격자에서 같은 기준에 댄다.
# (1) 해상도를 낮춰 가는 고전 솔버, (2) Lab 9.17의 FNO, (3) 한 인스턴스의 PINN(흐름 함수–와도 혼합 꼴, 강제 초기조건), (4) 손익분기.
# 줄인 포트: PINN은 강제 초기조건 판 하나만(파이썬은 벌점 판도 돌린다), 열방정식의 비교 (5)는 Lab 9.15의 출력에서 읽으면 된다.
# lab_9_16.jl(자료)과 lab_9_17.jl(FNO)을 먼저 돌려야 한다.
include("ch9_common.jl")
include("pinn_common.jl")
include("ns2d.jl")
include("fno.jl")

const NUV, KF, NFP, KC = 1e-3, 5, 2048, 12
const SN = [4KF + 1, 64, 64, 64, 64, 1]; const NPN = n_params(SN)       # ψ 망과 a 망, 각각 21-64-64-64-64-1
const KK = 2pi .* (1:KF)

"주기 특성(강제 주기 경계조건)과 x, y 방향의 1·2계 도함수, 그리고 t 방향. x, y, t는 1 x N"
function pinn_features(x, y, t, T)
    z0 = zeros(2KF, length(x)); z1 = zero(x)
    sx, cx, sy, cy = sin.(KK .* x), cos.(KK .* x), sin.(KK .* y), cos.(KK .* y)
    return (z = vcat(sx, cx, sy, cy, t ./ T),
            zx = vcat(KK .* cx, -KK .* sx, z0, z1), zxx = vcat(-KK .^ 2 .* sx, -KK .^ 2 .* cx, z0, z1),
            zy = vcat(z0, KK .* cy, -KK .* sy, z1), zyy = vcat(z0, -KK .^ 2 .* sy, -KK .^ 2 .* cy, z1),
            et = vcat(zeros(4KF), 1 / T))
end
forcing_xy(x, y) = 0.1 .* (sin.(2pi .* (x .+ y)) .+ cos.(2pi .* (x .+ y)))

"w0의 잘린 Fourier 급수(|k| ≤ 12, 625개 모드)와 그 도함수를 임의의 점에서: (w0, w0_x, w0_y, ∇²w0). 정확한 값이고 AD가 필요 없다"
function w0_series(w0, x, y)
    k = collect(fftfreq(64, 64)); C = fft(w0) ./ 64^2                   # C[jy, jx]
    keep = [(abs(k[jy]) <= KC) && (abs(k[jx]) <= KC) for jy in 1:64, jx in 1:64]
    kx = 2pi .* [k[jx] for jy in 1:64, jx in 1:64][keep]; ky = 2pi .* [k[jy] for jy in 1:64, jx in 1:64][keep]; c = C[keep]
    E = exp.(im .* (vec(x) .* kx' .+ vec(y) .* ky'))                    # 점 x 모드
    ev(v) = reshape(real.(E * v), 1, :)
    return ev(c), ev(im .* kx .* c), ev(im .* ky .* c), ev(-(kx .^ 2 .+ ky .^ 2) .* c)
end

"두 잔차: r1 = w_t + ψ_y w_x - ψ_x w_y - ν ∇²w - f,  r2 = ∇²ψ + w.  ansatz w = w0(x, y) + (t/T) a(x, y, t)"
function ns_residuals(θ, F, x, y, t, T, w0p)
    θψ, θa = θ[1:NPN], θ[NPN+1:2NPN]
    _, ψx, ψxx = jet2(SN, θψ, F.z, F.zx, F.zxx); _, ψy, ψyy = jet2(SN, θψ, F.z, F.zy, F.zyy)
    a, ax, axx = jet2(SN, θa, F.z, F.zx, F.zxx); _, ay, ayy = jet2(SN, θa, F.z, F.zy, F.zyy); _, at = jet1(SN, θa, F.z, F.et)
    tau = t ./ T
    w = w0p[1] .+ tau .* a
    wx, wy, wt = w0p[2] .+ tau .* ax, w0p[3] .+ tau .* ay, a ./ T .+ tau .* at
    lapw = w0p[4] .+ tau .* (axx .+ ayy)
    return wt .+ ψy .* wx .- ψx .* wy .- NUV .* lapw .- forcing_xy(x, y), ψxx .+ ψyy .+ w
end

"한 인스턴스(w0 → t ∈ [0, T])의 PINN. 예측 (64, 64, T + 1)과 걸린 시간을 돌려준다"
function pinn_instance(w0; T = 5.0, n_adam = 2000, n_lbfgs = 300, seed = 0)
    rng = Xoshiro(seed); x, y, t = rand(rng, 1, NFP), rand(rng, 1, NFP), T .* rand(rng, 1, NFP)   # 고정된 배치점
    F = pinn_features(x, y, t, T); w0p = w0_series(w0, x, y)
    θ = vcat(init_mlp(SN; seed)[3], init_mlp(SN; seed = seed + 1)[3])
    function loss(θ)
        r1, r2 = ns_residuals(θ, F, x, y, t, T, w0p)
        return mean(abs2, r1) + mean(abs2, r2)
    end
    g = (0:63) ./ 64; X0 = reshape([xx for yy in g, xx in g], 1, :); Y0 = reshape([yy for yy in g, xx in g], 1, :)
    w0g = w0_series(w0, X0, Y0)[1]
    predict(θ, tq) = reshape(w0g .+ (tq / T) .* mlp_value(SN, θ[NPN+1:2NPN], pinn_features(X0, Y0, fill(Float64(tq), size(X0)), T).z), 64, 64)
    t0 = time()
    θ = train_adam(loss, θ; iters = n_adam, lr = 2e-3, sched = it -> cosine_lr(2e-3, it, n_adam; eta_min = 1e-4))
    θ = train_lbfgs(loss, θ; iters = n_lbfgs)
    r1, r2 = ns_residuals(θ, F, x, y, t, T, w0p)
    @printf("    PINN (hard IC, two MLPs 21-64x4-1, %d parameters, %d fixed points): loss PDE %.2e, Poisson %.2e; truncation error of the w0 series %.1e\n",
            length(θ), NFP, mean(abs2, r1), mean(abs2, r2), rel_l2(reshape(w0g, 64, 64), w0))
    return stack([predict(θ, tq) for tq in 0:round(Int, T)]; dims = 3), time() - t0
end

function main(; T = 20)
    d = deserialize(joinpath(CH9_OUT, "95_ns.jls")); w0 = d.w0[:, :, 201:208]; test = Float64.(d.w[:, :, :, 201:208])
    t0 = time()
    ref = spectral_resample(trajectory(NS2D(128; dt = 2.5e-3), spectral_resample(w0, 128), T, 400), 64)
    @printf("reference 128^2, dt = 2.5e-3: %.0f s for 8 samples to T = %d\n", time() - t0, T)
    err(a, t) = mean(rel2d(a[:, :, t+1, :], ref[:, :, t+1, :]))
    println("\n(1) classical solver at decreasing resolution (query: w0 -> w(t), error on the 64^2 grid)")
    println("    N    dt       err t=1    err t=5    err t=20    ms per unit time (1 sample)")
    rows = []
    for (N, dt) in ((64, 1e-2), (48, 1e-2), (32, 2e-2), (24, 2e-2), (16, 4e-2), (12, 4e-2), (8, 4e-2))
        sol = NS2D(N; dt); n1 = round(Int, 1 / dt)
        tr = spectral_resample(trajectory(sol, spectral_resample(w0, N), T, n1), 64)
        w1 = spectral_resample(w0[:, :, 1:1], N); ms = 1e3 * best_time(() -> advance(sol, w1, n1))
        push!(rows, (N, err(tr, 20), ms))
        @printf("    %-3d  %.0e    %.2e   %.2e   %.2e    %6.2f\n", N, dt, err(tr, 1), err(tr, 5), err(tr, 20), ms)
    end
    println("\n(2) FNO-2D (Lab 9.17), autoregressive from w0")
    m = deserialize(joinpath(CH9_OUT, "95_fno_ns.jls")); p, sigma = m.p, m.sigma
    out = [Float32.(w0 ./ sigma)]
    for _ in 1:T
        push!(out, fno2d(p, out[end]))
    end
    pred = Float64.(stack(out; dims = 3)) .* sigma; x1 = out[1][:, :, 1:1]
    ms_fno = 1e3 * best_time(() -> fno2d(p, x1); rep = 20)
    @printf("    err t=1: %.2e, t=5: %.2e, t=20: %.2e;  %.2f ms per unit time (1 sample)\n", err(pred, 1), err(pred, 5), err(pred, 20), ms_fno)
    gen_s, train_s = d.seconds, m.seconds
    @printf("    one-off cost: data generation %.0f s (240 trajectories) + training %.0f s = %.0f s\n", gen_s, train_s, gen_s + train_s)

    println("\n(3) PINN for one instance (test trajectory 0), horizon T = 5")
    pp, cpu = pinn_instance(w0[:, :, 1])
    println("    hard IC: err t=1..5: ", join([@sprintf("%.3f", rel_l2(pp[:, :, t+1], ref[:, :, t+1, 1])) for t in 1:5], " "), @sprintf(";  %.0f s for this one instance", cpu))
    @printf("    same instance: FNO err t=1: %.3f, t=5: %.3f; 64^2 solver t=5: %.1e;  'nothing happens' (w = w0) t=1: %.3f, t=5: %.3f\n",
            rel_l2(pred[:, :, 2, 1], ref[:, :, 2, 1]), rel_l2(pred[:, :, 6, 1], ref[:, :, 6, 1]), rel_l2(test[:, :, 6, 1], ref[:, :, 6, 1]),
            rel_l2(w0[:, :, 1], ref[:, :, 2, 1]), rel_l2(w0[:, :, 1], ref[:, :, 6, 1]))

    println("\n(4) break-even number of queries (one query = one initial condition advanced to T = 20)")
    q_fno = T * ms_fno / 1e3; e_fno = err(pred, 20)
    for (N, e20, ms) in rows
        q_sol = T * ms / 1e3
        if q_sol > q_fno
            @printf("    vs solver %d^2 (err %.1e): %7.1f ms vs FNO %.1f ms per query -> speed-up %5.1fx, break-even N* = %.0f queries\n",
                    N, e20, 1e3 * q_sol, 1e3 * q_fno, q_sol / q_fno, (gen_s + train_s) / (q_sol - q_fno))
        else
            @printf("    vs solver %d^2 (err %.1e): %7.1f ms vs FNO %.1f ms per query -> the solver is %.1fx faster; no break-even\n", N, e20, 1e3 * q_sol, 1e3 * q_fno, q_fno / q_sol)
        end
    end
    ok = [r for r in rows if r[2] <= e_fno]; best = ok[argmin([r[3] for r in ok])]
    @printf("    matched accuracy at t = 20 (FNO error %.2e): coarsest solver at least as accurate is %d^2 (error %.1e, %.2f ms per unit time vs FNO %.2f ms)\n",
            e_fno, best[1], best[2], best[3], ms_fno)
end
main()
