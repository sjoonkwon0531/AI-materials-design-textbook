# lab_9_11.jl — Lab 9.11: KS 방정식의 자기회귀 롤아웃, 학습된 증폭인자, 스펙트럼, 질량 (labs/93_rollout_train.py, 93_rollout_eval.py)
# 한 걸음 FNO u(t) ↦ u(t + 1) (L = 64, N = 128, 솔버 걸음 0.025). 같은 사전학습에서 출발해 같은 갱신 수만큼 미세조정한다:
#   onestep(대조), noise(입력 잡음), push(자기 예측을 입력으로, 기울기는 끊고), unroll(4걸음을 펼쳐 역전파), cons(증분의 k = 0 모드 제거).
# 줄인 포트: 평가는 롤아웃 오차·유효 시간·발산 수·평균 표류·고파수 진폭과, u = 0에서의 정확한 von Neumann 인자까지.
# (쌍둥이 실험, 오차 스펙트럼의 시간 변화, 경험적 모드별 이득은 뺐다.)
include("ch9_common.jl")
include("spectral1d.jl")
include("fno.jl")
using NPZ

const FROM_PYTHON = false
const LDOM, N, NTR, NS = 64.0, 128, 64, 200
const KMAX, WIDTH, BATCH, PRE_EP, FT_UPD, KU = 24, 32, 64, 6, 600, 4
const TRAJ = FROM_PYTHON ? revdims(npzread(joinpath(CH9_DATA, "93_ks.npz"))["traj"]) : Float32.(ks_dataset())   # (128, 401, 80)
const X = TRAJ[:, 1:NS+1, 1:NTR]; const TE = TRAJ[:, :, NTR+1:end]
ks_step(p, u; cons = false) = fno1d(p, u; residual = true, conserve = cons)                  # u: (1, N, B)

"u(t), u(t+1), …, u(t+K): 각각 (1, N, BATCH)"
function batch(rng, K)
    i = rand(rng, 1:NTR, BATCH); n = rand(rng, 1:NS-K+1, BATCH)
    return [reshape(reduce(hcat, [X[:, n[b]+j, i[b]] for b in 1:BATCH]), 1, N, BATCH) for j in 0:K]
end
"기울기를 흘리지 않는 한 걸음 (pushforward의 '자기 예측' 입력)"
nograd_step(p, v) = ks_step(p, v)
ChainRulesCore.@non_differentiable nograd_step(p::Any, v::Any)

loss_onestep(p, u) = rel_loss(ks_step(p, u[1]), u[2])
loss_cons(p, u) = rel_loss(ks_step(p, u[1]; cons = true), u[2])
function loss_push(p, u)                                               # u = [u(t), …, u(t+k+1)], k ∈ {0, 1, 2}
    v = u[1]
    for _ in 1:length(u)-2
        v = nograd_step(p, v)
    end
    return rel_loss(ks_step(p, v), u[end])
end
function loss_unroll(p, u)                                             # KU걸음을 펼치고 모든 걸음으로 역전파
    v = u[1]; L = 0.0f0
    for j in 1:KU
        v = ks_step(p, v); L += rel_loss(v, u[j+1]) / KU
    end
    return L
end
"(손실 함수, 미니배치를 뽑는 함수)"
function make_loss(mode, sigma)
    mode == :onestep && return loss_onestep, (rng -> batch(rng, 1))
    mode == :cons && return loss_cons, (rng -> batch(rng, 1))
    mode == :push && return loss_push, (rng -> batch(rng, rand(rng, 0:2) + 1))
    mode == :unroll && return loss_unroll, (rng -> batch(rng, KU))
    noisy(rng) = (u = batch(rng, 1); [u[1] .+ Float32(sigma) .* randn(rng, Float32, size(u[1])), u[2]])
    return loss_onestep, noisy                                         # :noise
end
train(p, mode, updates, lr, seed; sigma = 0.0) = fit_steps(p, make_loss(mode, sigma)...; steps = updates, lr, clip = 1.0, seed)
onestep_err(p; cons = false) = mean(rel_each(ks_step(p, reshape(TE[:, 1:end-1, :], 1, N, :); cons), reshape(TE[:, 2:end, :], 1, N, :)))

"자기회귀 롤아웃 (N, steps + 1, B). post는 걸음마다 거는 후처리(사영, 필터)"
function rollout(p, u0, steps; cons = false, post = identity)
    out = [u0]; u = reshape(u0, 1, N, :)
    for _ in 1:steps
        u = clamp.(replace(post(ks_step(p, u; cons)), NaN32 => 1.0f6), -1.0f6, 1.0f6)
        push!(out, dropdims(u; dims = 1))
    end
    return stack(out; dims = 2)
end
"상관계수가 0.9 아래로 처음 떨어지는 걸음 (궤적마다)"
function valid_time(P, Tt; th = 0.9)
    c(a, b) = (a = a .- mean(a); b = b .- mean(b); sum(a .* b) / (norm(a) * norm(b) + 1e-12))
    return [something(findfirst(j -> c(P[:, j, b], Tt[:, j, b]) < th, 1:size(Tt, 2)), size(Tt, 2)) - 1 for b in 1:size(Tt, 3)]
end
highk(U) = sqrt.(sum(abs2, rfft(Float64.(U), 1)[34:end, :, :]; dims = 1)) ./ N       # k > 32의 진폭

function main()
    pre_upd = PRE_EP * (NTR * NS ÷ BATCH); models = Dict{Symbol,Any}()
    @printf("KS one-step data: %d trajectories x %d steps; FNO width %d, kmax %d, %d parameters; pre-training %d updates, fine-tuning %d updates, batch %d\n",
            NTR, NS, WIDTH, KMAX, count_params(init_fno1d(Xoshiro(0); width = WIDTH, kmax = KMAX)), pre_upd, FT_UPD, BATCH)
    pre = Dict(name => train(init_fno1d(Xoshiro(0); width = WIDTH, kmax = KMAX), name, pre_upd, 2e-3, 1) for name in (:onestep, :cons))
    for name in (:onestep, :cons)
        @printf("pre-train [%s]: one-step test rel-L2 %.4f\n", name, onestep_err(pre[name]; cons = name == :cons))
    end
    sigma = 0.5 * onestep_err(pre[:onestep]) * std(X)                   # 잡음 ≈ 한 걸음 RMS 오차의 절반
    for name in (:onestep, :noise, :push, :unroll, :cons)
        start = name == :cons ? pre[:cons] : pre[:onestep]               # 같은 사전학습에서 출발
        models[name] = train(start, name, FT_UPD, 5e-4, 2; sigma)
        @printf("fine-tune [%-7s]: one-step test rel-L2 %.4f\n", name, onestep_err(models[name]; cons = name == :cons))
    end
    serialize(joinpath(CH9_OUT, "93_rollout_models.jls"), models)

    u0 = TE[:, 1, :]; m0 = mean(u0; dims = 1)
    variants = Any[(String(n), models[n], n == :cons, identity) for n in (:onestep, :noise, :push, :unroll, :cons)]
    push!(variants, ("onestep+P0", models[:onestep], false, u -> u .- mean(u; dims = 2) .+ reshape(m0, 1, 1, :)))      # 걸음마다 평균을 되돌린다
    push!(variants, ("onestep+filter", models[:onestep], false, u -> (uh = rfft(u, 2); uh[:, 34:end, :] .= 0; irfft(uh, N, 2))))   # k > 32를 지운다
    for (name, p, cons, post) in variants
        P = rollout(p, u0, 2000; cons, post); P4 = P[:, 1:401, :]
        err = vec(mean(sqrt.(sum(abs2, P4 .- TE; dims = 1) ./ sum(abs2, TE; dims = 1)); dims = 3))
        vt = valid_time(P4, TE); blow = count(b -> any(>(20), abs.(P[:, :, b])), 1:size(P, 3))
        drift = mean(abs.(vec(mean(P[:, 401, :]; dims = 1)) .- vec(m0))); hk = vec(mean(highk(P[:, 1001:end, :]); dims = 2))
        @printf("%-15s: err(1) %.4f err(10) %.3f | valid time %5.1f +/- %4.1f | blow-ups in 2000 steps %d/16 | mean drift(t=400) %.1e | high-k amplitude (k>32) median %.1e max %.1e\n",
                name, err[2], err[11], mean(vt), std(vt), blow, drift, median(hk), maximum(hk))
    end
    @printf("solver         : high-k amplitude (k>32) median over test trajectories %.1e\n", median(vec(mean(highk(TE); dims = 2))))

    # u = 0에서의 정확한 von Neumann 인자: 야코비안의 열 J e_j를 접선 전파 한 번으로 모두 구한다(배치 = 단위 벡터 N개)
    @printf("exact linear KS: max_q exp(q^2 - q^4) = %.3f at q = %.3f (k = %.1f)\n", exp(0.25), sqrt(0.5), sqrt(0.5) * LDOM / 2pi)
    for name in (:onestep, :unroll)
        _, dn = fno1d_jvp(models[name], zeros(Float32, 1, N, N), reshape(Matrix{Float32}(I, N, N), 1, N, N))
        J = Matrix{Float32}(I, N, N) .+ dropdims(dn; dims = 1)          # 잔차 연결: J = I + ∂net/∂u
        circ = maximum(maximum(abs, circshift(J[:, j], -(j - 1)) .- J[:, 1]) for j in 1:N)
        g0 = abs.(rfft(Float64.(J[:, 1])))
        @printf("[%s] Jacobian at u=0: circulant defect %.1e; |g(k)| for k>kmax: min %.4f max %.4f (flat); max_k |g| = %.3f at k=%d\n",
                name, circ, minimum(g0[KMAX+2:end]), maximum(g0[KMAX+2:end]), maximum(g0), argmax(g0) - 1)
    end
    u16 = Float64.(u0)
    t_net = best_time(() -> rollout(models[:onestep], u0, 100); rep = 3) / 100 / 16
    t_sol = best_time(() -> ks_solve(u16, LDOM, 100.0, 0.025); rep = 3) / 100 / 16
    @printf("cost per trajectory per unit time: FNO step %.3f ms | solver, 40 steps of 0.025: %.3f ms\n", 1e3 * t_net, 1e3 * t_sol)
end
main()
