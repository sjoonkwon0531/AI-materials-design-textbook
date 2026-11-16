# lab_9_13.jl — Lab 9.13: 방정식 족에서 사전학습하고, 처음 보는 항에 few-shot으로 적응하기
# (labs/94_common.py, 94_data.py, 94b_pretrain.py, 94c_fewshot.py, 94e_physics_ft.py)
# 사전학습 족: u_t + c u_x + β (u²/2)_x = ν u_xx (ρ = 0).  표적: 반응항 ρ (u - u³), ρ ~ U[1, 3]이 더해진 족.
# 줄인 포트. 솔버 검증과 자료 생성, 앙상블 4개의 사전학습(Lab 9.14가 쓴다)은 그대로 옮겼다. 적응은 K = 16, 64에서
# 부분집합 하나씩, 방법마다 파이썬 판이 검증 집합으로 고른 학습률을 그대로 쓴다(학습률 탐색은 뺐다).
# 라벨 없는 적응(잔차만으로 미세조정)은 400걸음만 돌린다. FROM_PYTHON = true면 파이썬이 저장한 data/94_*.npz를 읽는다.
include("family.jl")
using NPZ

const FROM_PYTHON = false
const WIDTH, MODES, DEPTH, STEPS, BS, LR, MEMBERS = 32, 16, 4, 3000, 32, 3e-3, 4

function verify_solver()
    rng = Xoshiro(0); n = 256; u0 = sample_ic(rng, 8, n); coef = sample_coeffs(rng, 8, :pre); k = 0:n÷2
    lin = copy(coef); lin[3:4, :] .= 0                                   # β = ρ = 0: 정확한 해는 Fourier 곱수
    exact = irfft(rfft(u0, 1) .* exp.(-coef[1:1, :] .* k .^ 2 .- im .* coef[2:2, :] .* k), n, 1)
    @printf("linear case (beta = rho = 0): max |u - exact| at T = 1: %.1e\n", maximum(abs, etdrk4(u0, lin)[:, end, :] .- exact))
    for kind in (:pre, :react, :lownu)
        cf = sample_coeffs(Xoshiro(1), 8, kind)
        U = [etdrk4(u0, cf; dt, n_save = 4)[:, end, :] for dt in (1e-2, 5e-3, 2.5e-3)]
        e1, e2 = maximum(abs, U[1] .- U[2]), maximum(abs, U[2] .- U[3])
        @printf("%-6s: |u(dt=1e-2) - u(5e-3)| = %.1e, |u(5e-3) - u(2.5e-3)| = %.1e (ratio %.1f)\n", kind, e1, e2, e1 / e2)
    end
end

"자료 묶음 (u = (128, 41, n), coef = (4, n)). Julia 솔버로 만들거나 파이썬 npz를 읽는다. 한 번 만든 것은 저장해 둔다"
function dataset(name, n, seed, kind; kw...)
    FROM_PYTHON && (d = npzread(joinpath(CH9_DATA, "94_$(name).npz")); return (u = revdims(d["u"]), coef = permutedims(d["coef"])))
    path = joinpath(CH9_OUT, "94_$(name).jls")
    isfile(path) && return deserialize(path)
    d, w = make_dataset(n, seed; kind, kw...)
    @printf("%-12s N = %4d  kind = %-6s max|u| = %.2f  %.1f s (%.1f ms per trajectory)\n", name, n, kind, maximum(abs, d.u), w, 1e3 * w / n)
    serialize(path, d); return d
end
subset(d, idx) = (u = d.u[:, :, idx], coef = d.coef[:, idx])
err(p, d) = 100 * mean(traj_error(p, d))

"적응 방법 넷. 같은 갱신 수(400)와 배치: scratch(처음부터), last(마지막 층만), lora(점별 선형층에 랭크 r 어댑터), full(전부)"
function adapt(method, pre, sub, seed, lr; steps = 400, r = 4)
    rng = Xoshiro(1000 + seed); batchfn = g -> sample_batch(g, sub, BS)
    if method == :scratch
        return fit_steps(init_tfno(rng; width = WIDTH, modes = MODES, depth = DEPTH), data_loss, batchfn; steps, lr, pct = 0.05, seed), count_params(pre)
    elseif method == :full
        return fit_steps(pre, data_loss, batchfn; steps, lr, pct = 0.05, seed), count_params(pre)
    elseif method == :last                                               # q2만 학습: 나머지는 닫힘 안의 상수
        withq2(q) = (lift = pre.lift, blocks = pre.blocks, q1 = pre.q1, q2 = q)
        q2 = fit_steps(pre.q2, (q, b) -> data_loss(withq2(q), b), batchfn; steps, lr, pct = 0.05, seed)
        return withq2(q2), count_params(pre.q2)
    end
    a = fit_steps(init_lora(rng, pre, r), (a, b) -> data_loss(with_lora(pre, a), b), batchfn; steps, lr, pct = 0.05, seed)
    return with_lora(pre, a), count_params(a)
end

"라벨 없는 적응: 방정식을 아므로 입력은 공짜다. 매 걸음 새 (u0, 계수, t)를 뽑아 사전학습 모형의 PDE 잔차를 줄인다"
function physics_finetune(pre; steps = 400, lr = 1e-3, seed = 0)
    batchfn(g) = (Float32.(sample_ic(g, BS, 128)), Float32.(sample_coeffs(g, BS, :react)), rand(g, Float32, BS))
    loss(p, (u0, coef, t)) = mean(abs2, pde_residual(p, u0, coef, t)[2])
    return fit_steps(pre, loss, batchfn; steps, lr, pct = 0.05, seed)
end

function main()
    verify_solver()
    tr = dataset("pre_train", 2000, 1, :pre); te = dataset("pre_test", 300, 3, :pre)
    low = dataset("lownu_test", 300, 4, :lownu); hk = dataset("highk_test", 300, 7, :pre; kmax = 10, amp = 1.5)
    pool = dataset("react_train", 256, 5, :react); rea = dataset("react_test", 300, 6, :react); dataset("pre_cal", 300, 2, :pre)
    @printf("pre-training set: %d trajectories x %d times x %d points\n", size(tr.u, 3), size(tr.u, 2) - 1, size(tr.u, 1))
    members = []
    for k in 0:MEMBERS-1
        path = joinpath(CH9_OUT, "94_tfno_member$(k).jls")
        if !isfile(path)
            t0 = time()
            p = train_tfno(init_tfno(Xoshiro(k); width = WIDTH, modes = MODES, depth = DEPTH), tr; steps = STEPS, bs = BS, lr = LR, seed = k, log = k == 0 ? 1000 : 0)
            serialize(path, (p = p, seconds = time() - t0))
        end
        m = deserialize(path); push!(members, m.p)
        @printf("member %d: %d params, %d steps, %.0f s | ID %.2f%% | low-nu %.2f%% | high-k %.2f%% | reaction %.2f%%\n",
                k, count_params(m.p), STEPS, m.seconds, err(m.p, te), err(m.p, low), err(m.p, hk), err(m.p, rea))
    end
    pre = members[1]
    # 구조 점검: 사전학습 족(ρ = 0)은 평균을 정확히 보존한다. 보존을 넣지 않은 FNO는 얼마나 지키는가
    pred = predict(pre, te, T_IDX); drift = abs.(dropdims(mean(pred; dims = 1); dims = 1) .- mean(te.u[:, 1:1, :]; dims = 1)[1, :, :])
    @printf("pre_test: FNO mean drift %.2e, max %.2e (the solver conserves the mean to rounding)\n", mean(drift), maximum(drift))

    @printf("pretrained member 0: ID test %.2f%% | zero-shot on the reaction family %.2f%%\n", err(pre, te), err(pre, rea))
    @printf("trivial baseline u(x, t) = u0(x): %.1f%%\n", 100 * mean(rel_each(repeat(rea.u[:, 1:1, :], 1, length(T_IDX), 1), rea.u[:, T_IDX, :])))
    best_lr = Dict((16, :scratch) => 1e-2, (16, :last) => 1e-2, (16, :lora) => 1e-2, (16, :full) => 3e-3,
                   (64, :scratch) => 1e-2, (64, :last) => 1e-2, (64, :lora) => 1e-2, (64, :full) => 1e-2)   # 파이썬 판의 검증 선택
    for K in (16, 64)
        @printf("--- K = %d labelled trajectories (%d space-time fields), 400 steps ---\n", K, 40K)
        sub = subset(pool, randperm(Xoshiro(10K), 224)[1:K])              # 학습 부분집합은 pool[1:224]에서
        for meth in (:scratch, :last, :lora, :full)
            t0 = time(); p, ntr = adapt(meth, pre, sub, 0, best_lr[(K, meth)])
            @printf("  %-8s trainable %6d  lr %.0e  target-test error %6.2f %%   ID-test error after adaptation %6.2f %%   (%.0f s)\n",
                    meth, ntr, best_lr[(K, meth)], err(p, rea), err(p, te), time() - t0)
        end
    end
    r0 = mean(residual_norms((u0, c, t) -> tfno_jvp(pre, u0, c, t), rea))
    @printf("pretrained (zero-shot): target-test error %.2f %%, rms residual %.3f\n", err(pre, rea), r0)
    t0 = time(); pf = physics_finetune(pre)
    @printf("  residual only, 400 steps (0 labels)    target-test error %6.2f %%   rms residual %.3f   (%.0f s)\n",
            err(pf, rea), mean(residual_norms((u0, c, t) -> tfno_jvp(pf, u0, c, t), rea)), time() - t0)
end
main()
