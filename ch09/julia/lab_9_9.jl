# lab_9_9.jl — Lab 9.9: DeepONet 대 FNO, 선형 디코더의 바닥과 자료 효율 (labs/93_deeponet.py)
# 같은 Burgers 연산자 u0 ↦ u(·, T), 센서 64개. DeepONet의 branch·trunk는 Lux.jl의 MLP다(평범한 순전파뿐이라 Lux를 그대로 쓴다).
#   G(u)(y) ≈ Σ_k b_k(u(x_1..x_m)) t_k(y) + b0,  trunk 입력은 주기 특성 [cos 2πjy, sin 2πjy], j = 1..4
# 줄인 포트: POD 바닥(결정론적), N = 1000에서의 DeepONet·POD-DeepONet·FNO, 함께 움직이는(Galilei) 좌표계에서의 DeepONet,
# 평행이동 등변성 오차. 파이썬 판의 학습 곡선(N = 250…2000)은 run_don(…, ntr)를 되풀이 부르면 된다.
include("ch9_common.jl")
include("spectral1d.jl")
include("fno.jl")
using Lux, NPZ

const FROM_PYTHON = false
const NF, S, T, NU = 512, 64, 0.25, 0.005

function load_data()
    FROM_PYTHON && (traj = revdims(npzread(joinpath(CH9_DATA, "93_burgers.npz"))["traj"]); return traj[:, 1, :], traj[:, end, :])
    traj = burgers_dataset(; n = 2200, N = NF, nu = NU, T)
    return traj[:, 1, :], traj[:, end, :]
end
"Galilei 좌표계: v0 = u0 - 평균, v(y, T) = u(y + 평균·T, T) - 평균 (고운 격자 위의 정확한 Fourier 이동)"
function comoving(A, U)
    m = mean(A; dims = 1); k = 2pi .* (0:NF÷2)
    V = irfft(rfft(Float64.(U), 1) .* exp.(im .* k .* m .* T), NF, 1) .- m
    return A .- m, Float32.(V)
end

"PyTorch nn.Linear 분포의 Lux Dense (Float32)"
tinit(nin) = (rng, dims...) -> (rand(rng, Float32, dims...) .* 2.0f0 .- 1.0f0) ./ sqrt(Float32(nin))
tdense(nin, nout, act = identity) = Dense(nin => nout, act; init_weight = tinit(nin), init_bias = tinit(nin))
mlp(sizes; last_act = false) = Chain([tdense(sizes[i], sizes[i+1], (i < length(sizes) - 1 || last_act) ? gelu_exact : identity) for i in 1:length(sizes)-1]...)

const YQ = reshape(Float32.((0:S-1) ./ S), 1, :)                          # 질의점 (1, Q)
const TRUNK_IN = vcat([cos.(2.0f0 * Float32(pi) * j .* YQ) for j in 1:4]..., [sin.(2.0f0 * Float32(pi) * j .* YQ) for j in 1:4]...)   # (8, Q)

"학습한 DeepONet의 예측 (Q, B): trunk(Q개 점)ᵀ · branch(B개 함수) + b0 [+ 평균장]"
function don_apply(branch, trunk, stb, stt, ps, u; basis = nothing, meanfield = nothing)
    bm = first(branch(u, ps.branch, stb))                                 # (p, B)
    tm = basis === nothing ? first(trunk(TRUNK_IN, ps.trunk, stt)) : basis   # (p, Q)
    out = tm' * bm .+ ps.b0
    return meanfield === nothing ? out : out .+ meanfield
end

"DeepONet(pod = false) 또는 POD-DeepONet(학습 출력의 POD 모드 p개를 trunk로 고정)을 ntr개 함수로 학습. (시험 오차, 학습 손실, 매개변수 수, 예측 함수)"
function run_don(a, u, ntr, te; p = 64, pod = false, updates = 6000, seed = 0, width = 128)
    rng = Xoshiro(seed)
    branch = mlp([S, width, width, p]); trunk = mlp([8, width, width, p]; last_act = true)
    psb, stb = Lux.setup(rng, branch); pst, stt = Lux.setup(rng, trunk)
    basis = meanfield = nothing
    if pod
        meanfield = mean(u[:, 1:ntr]; dims = 2)
        basis = permutedims(svd(u[:, 1:ntr] .- meanfield).U[:, 1:p]) .* sqrt(Float32(S))     # (p, Q): 왼쪽 특이벡터가 공간 모드
    end
    ps = pod ? (branch = psb, b0 = zeros(Float32, 1)) : (branch = psb, trunk = pst, b0 = zeros(Float32, 1))
    apply(q, x) = don_apply(branch, trunk, stb, stt, q, x; basis, meanfield)
    bs = 50; ep = max(1, round(Int, updates * bs / ntr))
    q, hist = fit_epochs(ps, (q, idx) -> rel_loss(apply(q, a[:, idx]), u[:, idx]), ntr; epochs = ep, batch = bs, lr = 2e-3, wd = 1e-5, seed)
    return mean(rel_each(apply(q, a[:, te]), u[:, te])), hist[end], count_params(ps), x -> apply(q, x)
end

function run_fno(a, u, ntr, te; updates = 2500, seed = 0)
    bs = 20; ep = max(1, round(Int, updates * bs / ntr)); a3, u3 = reshape(a, 1, S, :), reshape(u, 1, S, :)
    p, hist = fit_epochs(init_fno1d(Xoshiro(seed)), (p, idx) -> rel_loss(fno1d(p, a3[:, :, idx]), u3[:, :, idx]), ntr; epochs = ep, batch = bs, lr = 3e-3, wd = 1e-5, seed)
    return mean(rel_each(fno1d(p, a3[:, :, te]), u3[:, :, te])), hist[end]
end

function main()
    A, U = load_data(); te = 2001:2200; st = NF ÷ S
    A0, V = comoving(A, U)
    frames = (lab = (A[1:st:end, :], U[1:st:end, :]), comoving = (A0[1:st:end, :], V[1:st:end, :]))
    for (name, (a, u)) in pairs(frames)                                   # (1) POD 바닥: 기저 p개의 어떤 선형 디코더도 넘지 못하는 오차
        mf = mean(u[:, 1:1000]; dims = 2); Um = svd(u[:, 1:1000] .- mf).U
        fl = [mean(sqrt.(sum(abs2, (u[:, te] .- mf) .- Um[:, 1:p] * (Um[:, 1:p]' * (u[:, te] .- mf)); dims = 1)) ./ sqrt.(sum(abs2, u[:, te]; dims = 1))) for p in (2, 4, 8, 16, 32, 64)]
        @printf("[%-8s] POD floor (test rel-L2 of the best p-term linear reconstruction): %s\n", name, join([@sprintf("p=%d: %.4f", p, f) for (p, f) in zip((2, 4, 8, 16, 32, 64), fl)], "  "))
    end
    ratio = mean(sqrt.(sum(abs2, frames.comoving[2][:, te]; dims = 1)) ./ sqrt.(sum(abs2, frames.lab[2][:, te]; dims = 1)))
    @printf("norm ratio ||v(T)|| / ||u(T)|| (test mean) = %.3f\n", ratio)
    a, u = frames.lab
    e, tr, np_, don = run_don(a, u, 1000, te);            @printf("[lab, N=1000] deeponet      (%6d params): test rel-L2 %.4f (train %.4f)\n", np_, e, tr)
    e2, tr2, np2, _ = run_don(a, u, 1000, te; p = 32, pod = true); @printf("[lab, N=1000] pod_deeponet  (%6d params): test rel-L2 %.4f (train %.4f)\n", np2, e2, tr2)
    e3, tr3 = run_fno(a, u, 1000, te);                    @printf("[lab, N=1000] fno           (%6d params): test rel-L2 %.4f (train %.4f)\n", count_params(init_fno1d(Xoshiro(0))), e3, tr3)
    ac, uc = frames.comoving
    e4, tr4, _, _ = run_don(ac, uc, 1000, te);            @printf("[comoving, N=1000] deeponet: test rel-L2 %.4f (train %.4f)  -> relative to ||u(T)||: %.4f\n", e4, tr4, e4 * ratio)
    eq = mean(rel_each(don(circshift(a[:, te], (5, 0))), circshift(don(a[:, te]), (5, 0))))
    @printf("DeepONet translation-equivariance error (shift 5/64): %.3f\n", eq)
end
main()
