# lab_9_17.jl — Lab 9.17: FNO-2D, 단위 시간 흐름 사상 w(t) ↦ w(t + 1)을 학습하고 20스텝을 굴린다 (labs/95_fno2d.py, 95_fno_ns.py)
# 무거운 Lab이다(파이썬 판: Adam 2,000걸음, CPU 약 270 s). 자료는 lab_9_16.jl이 만든 models/95_ns.jls
# (FROM_PYTHON = true면 data/95_ns_nu1e-3_64.npz). 모형과 손실, 학습 루프, 롤아웃 평가의 핵심을 옮겼다.
# 줄인 포트: 가장자리 고리의 오차와 스펙트럼 표의 일부는 뺐다.
include("ch9_common.jl")
include("ns2d.jl")
include("fno.jl")
using NPZ

const FROM_PYTHON = false
const EPOCHS, BATCH, LR, NTRAIN = 4, 8, 2e-3, 200
"(64, 64, 21, 240) Float32"
load_w() = FROM_PYTHON ? revdims(npzread(joinpath(CH9_DATA, "95_ns_nu1e-3_64.npz"))["w"]) : deserialize(joinpath(CH9_OUT, "95_ns.jls")).w

"자기회귀 롤아웃 (물리 단위). w0: (64, 64, B) → (64, 64, nsteps + 1, B)"
function rollout(p, w0, nsteps, sigma)
    out = [Float32.(w0 ./ sigma)]
    for _ in 1:nsteps
        push!(out, fno2d(p, out[end]))
    end
    return stack(out; dims = 3) .* sigma
end
"교사 강제 한 걸음 오차 (T, B): 참 w(t) → w(t + 1)"
function one_step_error(p, traj, sigma; bs = 40)
    x = reshape(traj[:, :, 1:end-1, :], 64, 64, :) ./ sigma
    pr = cat([fno2d(p, x[:, :, i:min(i + bs - 1, size(x, 3))]) for i in 1:bs:size(x, 3)]...; dims = 3) .* sigma
    return rel2d(reshape(pr, size(traj[:, :, 2:end, :])), traj[:, :, 2:end, :])
end
"각도 평균한 엔스트로피 스펙트럼 (껍질 |k| = 0..32), 표본 평균"
function enstrophy_spectrum(f)
    k = fftfreq(64, 64); kb = round.(Int, sqrt.(k .^ 2 .+ (k') .^ 2)); out = zeros(46)
    fh = dropdims(mean(abs2.(fft(Float64.(f), (1, 2))); dims = 3); dims = 3) ./ 64^4
    for i in eachindex(kb)
        out[kb[i]+1] += fh[i]
    end
    return out[1:33]
end

function main()
    w = load_w(); T = size(w, 3) - 1
    train, test = w[:, :, :, 1:NTRAIN], w[:, :, :, NTRAIN+1:end]
    sigma = std(train; corrected = false)                               # 전역 척도 하나: 사상이 척도 불변이 아니다
    x_tr = reshape(train[:, :, 1:end-1, :], 64, 64, :) ./ sigma; y_tr = reshape(train[:, :, 2:end, :], 64, 64, :) ./ sigma
    @printf("train pairs %d, test trajectories %d, sigma = %.3f\n", size(x_tr, 3), size(test, 4), sigma)
    p0 = init_fno2d(Xoshiro(0); modes = 12, width = 20, depth = 4, proj = 64)
    @printf("FNO-2D: modes 12, width 20, 4 Fourier layers, %d real parameters\n", count_params(p0))
    t0 = time()
    p, hist = fit_epochs(p0, (p, idx) -> rel_loss(fno2d(p, x_tr[:, :, idx]), y_tr[:, :, idx]), size(x_tr, 3);
                         epochs = EPOCHS, batch = BATCH, lr = LR, drop_last = true)
    @printf("training: %.0f s (%d Adam steps, batch %d); train loss per epoch: %s\n", time() - t0, EPOCHS * (size(x_tr, 3) ÷ BATCH), BATCH, join([@sprintf("%.4f", h) for h in hist], " "))
    serialize(joinpath(CH9_OUT, "95_fno_ns.jls"), (p = p, sigma = sigma, seconds = time() - t0))

    e_one = one_step_error(p, test, sigma)                              # (T, B)
    pred = rollout(p, test[:, :, 1, :], T, sigma)
    e_roll = rel2d(pred[:, :, 2:end, :], test[:, :, 2:end, :])
    e_pers1 = rel2d(test[:, :, 1:end-1, :], test[:, :, 2:end, :])       # 지속 예보, 한 걸음
    e_pers0 = rel2d(repeat(test[:, :, 1:1, :], 1, 1, T, 1), test[:, :, 2:end, :])
    println("  t      one-step     rollout      persistence(1 step)   persistence(from t=0)")
    for t in (1, 2, 5, 10, 15, 20)
        @printf("  %2d     %.4f       %.4f       %.4f                %.4f\n", t, mean(e_one[t, :]), mean(e_roll[t, :]), mean(e_pers1[t, :]), mean(e_pers0[t, :]))
    end
    @printf("mean over t = 1..20: one-step %.4f, rollout %.4f; rollout at t = 20: %.4f +/- %.4f (s.e., %d samples), worst sample %.4f\n",
            mean(e_one), mean(e_roll), mean(e_roll[end, :]), std(e_roll[end, :]; corrected = false) / sqrt(size(test, 4)), size(test, 4), maximum(e_roll[end, :]))
    st, sp = enstrophy_spectrum(test[:, :, end, :]), enstrophy_spectrum(pred[:, :, end, :])
    println("enstrophy spectrum at t = 20 (shells |k| = 1, 2, 4, 8, 16):")
    println("  solver : ", join([@sprintf("%.2e", st[k+1]) for k in (1, 2, 4, 8, 16)], " "))
    println("  FNO    : ", join([@sprintf("%.2e", sp[k+1]) for k in (1, 2, 4, 8, 16)], " "))
    @printf("  share of the solver enstrophy above |k| = 12: %.2e; FNO: %.2e\n", sum(st[14:end]) / sum(st), sum(sp[14:end]) / sum(sp))
    @printf("mean vorticity (conserved, = 0): max |<w>| over the FNO rollouts = %.2e, solver %.2e\n",
            maximum(abs, mean(pred; dims = (1, 2))), maximum(abs, mean(test; dims = (1, 2))))
end
main()
