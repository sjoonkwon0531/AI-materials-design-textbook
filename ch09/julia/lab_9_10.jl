# lab_9_10.jl — Lab 9.10: PINO, 라벨이 적을 때의 잔차 손실, 그리고 인스턴스별 미세조정 (labs/93_pino.py)
# Burgers, ν = 0.02, T = 0.25. 시간 조건부 FNO-1D: G_θ(u0, t)(x) = u0(x) + (t/T) N_θ(u0, t/T)(x) (강제 초기조건).
# 잔차 r = u_t + u u_x - ν u_xx: x 도함수는 스펙트럴, t 도함수는 fno1d_jvp(시간 채널 방향의 접선을 층마다 전파)로 정확히.
# 줄인 포트: 연산자 학습은 (data, 25), (both, 25), (pde, 0) 셋, 인스턴스별 미세조정은 네 출발점(data25, both25, pde0, 무작위 초기화) 모두에서.
include("ch9_common.jl")
include("spectral1d.jl")
include("fno.jl")

const NU, T, NF, S_DATA, S_PDE = 0.02, 0.25, 256, 64, 128
"(u, ∂u/∂t). a는 (1, N, B), t는 (B)"
function G_jvp(p, a, t)
    N, B = size(a, 2), size(a, 3); F = eltype(a)
    tau = repeat(reshape(t ./ F(T), 1, 1, B), 1, N, 1)
    n, dn = fno1d_jvp(p, vcat(a, tau), vcat(zero(a), ones(F, 1, N, B)))    # 입력의 τ 채널 방향 접선
    return a .+ tau .* n, (n .+ tau .* dn) ./ F(T)                          # ∂/∂t = (1/T) ∂/∂τ
end
G(p, a, t) = (tau = repeat(reshape(t ./ eltype(a)(T), 1, 1, :), 1, size(a, 2), 1); a .+ tau .* fno1d_net(p, vcat(a, tau)))
"스펙트럴 기호 i k와 -k² (1, nk, 1). 상수"
dsym(N, F) = (k = F(2pi) .* F.(0:N÷2); (reshape(im .* k, 1, :, 1), reshape(-k .^ 2, 1, :, 1)))
ChainRulesCore.@non_differentiable dsym(N::Any, F::Any)
"잔차 u_t + u u_x - ν u_xx (u, ut는 (1, N, B))"
function burgers_res(u, ut)
    N = size(u, 2); ik, mk2 = dsym(N, eltype(u)); uh = rfft(u, 2)
    return ut .+ u .* irfft(ik .* uh, N, 2) .- eltype(u)(NU) .* irfft(mk2 .* uh, N, 2)
end
residual(p, a, t) = burgers_res(G_jvp(p, a, t)...)

function main()
    # 잔차 루틴의 점검: 정확한 Cole–Hopf 해를 넣으면 0이어야 한다 (Float64)
    x = reshape(collect(0:127) ./ 128, 1, :, 1); ts = reshape([0.05, 0.1, 0.2, 0.25], 1, 1, :)
    e = exp.(-4pi^2 * NU .* ts)
    u = 4pi * NU .* e .* sin.(2pi .* x) ./ (1.5 .+ e .* cos.(2pi .* x))
    ut = -4pi^2 * NU .* (4pi * NU .* e .* sin.(2pi .* x) .* 1.5) ./ (1.5 .+ e .* cos.(2pi .* x)) .^ 2
    @printf("residual routine on the exact Cole-Hopf solution: max|r| = %.1e\n", maximum(abs, burgers_res(u, ut)))

    # 자료: 라벨 있는 궤적 100개 + 시험 200개, 라벨 없는 초기조건 4000개
    rng = Xoshiro(95); a0 = grf(rng, 300, NF)
    t0 = time(); traj = Float32.(burgers_solve(a0, NU, T, 1e-3; save_every = 25)); t_solver = (time() - t0) / 300   # (256, 11, 300)
    A_unl = Float32.(grf(rng, 4000, NF)); tgrid = Float32.(range(0, T; length = 11)); TE = 101:300
    d64(X) = reshape(X[1:NF÷S_DATA:end, :], 1, S_DATA, :); d128(X) = reshape(X[1:NF÷S_PDE:end, :], 1, S_PDE, :)
    test(p) = rel_each(G(p, d64(traj[:, 1, TE]), fill(Float32(T), 200)), d64(traj[:, end, TE]))
    res_mse(p) = mean(abs2, residual(p, d128(traj[:, 1, TE[1:64]]), Float32.(range(0.02, 1; length = 64) .* T)))
    @printf("Burgers nu=%g, T=%g: solver N=%d, dt=1e-3 (%.2f ms per trajectory); data loss at s=%d, residual at s=%d\n", NU, T, NF, 1e3 * t_solver, S_DATA, S_PDE)

    # mode ∈ (:data, :both, :pde). 라벨 손실은 s = 64의 스냅숏, 잔차 손실은 s = 128의 라벨 없는 입력과 무작위 시각
    function train(mode, nlab; updates = 1500, lam = 0.03f0, seed = 0)
        function batch(rng)
            i, j = rand(rng, 1:max(nlab, 1), 16), rand(rng, 2:11, 16)
            lab = (d64(traj[:, 1, i]), tgrid[j], reshape(reduce(hcat, [traj[1:NF÷S_DATA:end, j[b], i[b]] for b in 1:16]), 1, S_DATA, :))
            return lab, (d128(A_unl[:, rand(rng, 1:4000, 16)]), rand(rng, Float32, 16) .* Float32(T))
        end
        function loss(p, (lab, unl))
            L = 0.0f0
            if mode != :pde
                L += rel_loss(G(p, lab[1], lab[2]), lab[3])
            end
            if mode != :data
                L += lam * mean(abs2, residual(p, unl[1], unl[2]))
            end
            return L
        end
        t0 = time()
        p = fit_steps(init_fno1d(Xoshiro(seed); in_ch = 2), loss, batch; steps = updates, lr = 2e-3, clip = 1.0, seed)
        return p, time() - t0
    end

    # 인스턴스별 미세조정: 시험 입력 하나의 잔차를 고정된 nt개 시각에서 줄인다 (Adam + 코사인). 라벨은 오차를 재는 데만 쓴다
    function finetune(p0, j; steps = 200, lr = 3e-4, nt = 8)
        a = repeat(d128(traj[:, 1, 100+j:100+j]), 1, 1, nt); t = Float32.(((1:nt) .- 0.5) ./ nt .* T)
        a64 = d64(traj[:, 1, 100+j:100+j]); tru = traj[1:NF÷S_DATA:end, end, 100+j]
        err(p) = rel_l2(vec(G(p, a64, [Float32(T)])), tru)
        p = fit_steps(p0, (p, _) -> mean(abs2, residual(p, a, t)), rng -> nothing; steps, lr, sched = s -> cosine_lr(lr, s, steps))
        return err(p0), err(p), mean(abs2, residual(p0, a, t)), mean(abs2, residual(p, a, t))
    end

    nets = Dict{String,Any}()
    for (mode, nlab) in ((:data, 25), (:both, 25), (:pde, 0))
        p, w = train(mode, nlab); e = test(p); nets["$(mode)$(nlab)"] = p
        @printf("operator [%-4s, %3d labelled]: test rel-L2 %.4f (median %.4f)  residual MSE %8.3f   (%.0f s)\n", mode, nlab, mean(e), median(e), res_mse(p), w)
    end
    nets["random"] = init_fno1d(Xoshiro(1); in_ch = 2)
    for key in ("data25", "both25", "pde0", "random")
        R = [finetune(nets[key], j; lr = key == "random" ? 2e-3 : 3e-4) for j in 1:6]
        @printf("fine-tune from [%-6s] (6 instances, 200 steps): rel-L2 %.4f -> %.4f   residual MSE %.2f -> %.3f\n", key,
                mean(r[1] for r in R), mean(r[2] for r in R), mean(r[3] for r in R), mean(r[4] for r in R))
    end
end
main()
