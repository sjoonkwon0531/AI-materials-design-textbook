# lab_9_20.jl — Lab 9.20: Cahn–Hilliard의 FNO 대리 모델, 세 모델, 세 가지 구조 점검, 분포 밖, 비용
# (labs/96_fno_ch_train.py, 96_fno_ch_eval.py)
# 사상 c(t) ↦ c(t + 2.5) (솔버 50걸음)를 160개 궤적(쌍 3,200개)에서 배운다. 세 모델:
#   free(보통 FNO, 한 걸음 손실), cons(증분의 k = 0 모드 제거 → 질량이 구조로 보존), cons_pf(cons + pushforward 손실).
# 무거운 Lab이다(파이썬 판: 모델마다 CPU 300–500 s). 모형·손실·학습 루프와 평가의 핵심(한 걸음·롤아웃 오차, 질량, 자유에너지,
# 성장 법칙, 대칭, 분포 밖, 비용)을 옮겼다. 줄인 포트: 급랭 걸음까지 맡긴 롤아웃과 '큰 dt의 솔버' 표는 뺐다.
# lab_9_19.jl(자료)을 먼저 돌려야 한다. 추론은 Float64로 한다(구조 점검의 숫자가 반올림에 묻히지 않게).
include("ch9_common.jl")
include("ch2d.jl")
include("fno.jl")

const EPOCHS, BATCH, LR, MODES, WIDTH = 5, 8, 2e-3, 16, 20
const D = deserialize(joinpath(CH9_OUT, "96_ch.jls"))
const NAMES = Dict(:free => "FNO, one-step loss", :cons => "FNO + zero mode, one-step loss", :cons_pf => "FNO + zero mode, pushforward loss")

"기울기를 끊은 자기 예측 (pushforward의 둘째 항의 입력)"
detached(p, x, cons) = fno2d(p, x; conserve = cons)
ChainRulesCore.@non_differentiable detached(p::Any, x::Any, cons::Any)

function train_model(tag; seed = 0)
    cons = tag != :free; pf = tag == :cons_pf; tr = D.train
    nx = pf ? 19 : 20                                                   # pf는 세 쌍둥이 (c_t, c_{t+1}, c_{t+2})
    x = reshape(tr[:, :, 1:nx, :], 64, 64, :); y = reshape(tr[:, :, 2:nx+1, :], 64, 64, :)
    z = pf ? reshape(tr[:, :, 3:nx+2, :], 64, 64, :) : y
    function loss(p, idx)
        pred = fno2d(p, x[:, :, idx]; conserve = cons)
        L = rel_loss(pred, y[:, :, idx])
        if pf                                                           # 모형이 자기 출력을 입력으로 본다. 첫 걸음으로는 기울기를 흘리지 않는다
            L += rel_loss(fno2d(p, detached(p, x[:, :, idx], cons); conserve = cons), z[:, :, idx])
        end
        return L
    end
    p0 = init_fno2d(Xoshiro(seed); modes = MODES, width = WIDTH, depth = 4, proj = 64)
    @printf("[%s] FNO-2D modes %d, width %d, 4 layers, %d real parameters; %d training %s\n", tag, MODES, WIDTH, count_params(p0), size(x, 3), pf ? "triples" : "pairs")
    t0 = time()
    p, hist = fit_epochs(p0, loss, size(x, 3); epochs = EPOCHS, batch = BATCH, lr = LR, drop_last = true, seed)
    @printf("[%s] training: %.0f s; train loss per epoch: %s\n", tag, time() - t0, join([@sprintf("%.4f", h) for h in hist], " "))
    serialize(joinpath(CH9_OUT, "96_fno_ch_$(tag).jls"), (p = p, seconds = time() - t0))
end

fno_apply(p, x, cons; bs = 20) = cat([fno2d(p, x[:, :, i:min(i + bs - 1, size(x, 3))]; conserve = cons) for i in 1:bs:size(x, 3)]...; dims = 3)
"교사 강제 한 걸음: 참 c(t) → c(t + Δt). (64, 64, 20, B)"
one_step(p, traj, cons) = reshape(fno_apply(p, reshape(traj[:, :, 1:end-1, :], 64, 64, :), cons), 64, 64, :, size(traj, 4))
function rollout(p, c0, n, cons)
    out = [c0]
    for _ in 1:n
        push!(out, fno_apply(p, out[end], cons))
    end
    return stack(out; dims = 3)
end
rel4(a, b) = dropdims(sqrt.(sum(abs2, a .- b; dims = (1, 2)) ./ sum(abs2, b; dims = (1, 2))); dims = (1, 2))    # (시각, 궤적)
function diagnostics(sol, traj)
    F = reduce(hcat, [free_energy(sol, traj[:, :, i, :]) for i in 1:size(traj, 3)])'
    L = reduce(hcat, [char_length(sol, traj[:, :, i, :]) for i in 1:size(traj, 3)])'
    return F, L
end

function evaluate()
    sol = CH2D(; N = 64); test = Float64.(D.test); ood = Float64.(D.ood); nrec = size(test, 3) - 1
    t1 = 2.5 .* (1:nrec)                                                # c(2.5)에서 시작한 롤아웃의 시각
    Fs, Ls = diagnostics(sol, test[:, :, 2:end, :])
    pers = rel4(test[:, :, 1:end-1, :], test[:, :, 2:end, :])
    println("=== in-distribution test set: 40 trajectories, |m| <= 0.2 ===")
    @printf("baselines:  persistence (one step) 0->2.5: %.3f | 2.5->5: %.3f | 47.5->50: %.3f | mean of steps 2..20: %.3f\n", mean(pers[1, :]), mean(pers[2, :]), mean(pers[20, :]), mean(pers[2:end, :]))
    @printf("solver:     F(2.5) %.3f -> F(50) %.3f, increases %d / %d; L(25) %.3f, L(50) %.3f, slope on 5 <= t <= 50: %.3f\n",
            mean(Fs[1, :]), mean(Fs[end, :]), count(>(0), diff(Fs; dims = 1)), length(Fs) - size(Fs, 2), mean(Ls[10, :]), mean(Ls[end, :]), growth_slope(t1, vec(mean(Ls; dims = 2))))
    models = Dict(tag => deserialize(joinpath(CH9_OUT, "96_fno_ch_$(tag).jls")) for tag in (:free, :cons, :cons_pf))
    for tag in (:free, :cons, :cons_pf)
        p = to64(models[tag].p); cons = tag != :free
        p1 = one_step(p, test, cons); e1 = rel4(p1, test[:, :, 2:end, :])
        r = rollout(p, test[:, :, 2, :], nrec - 1, cons); er = rel4(r[:, :, 2:end, :], test[:, :, 3:end, :])
        F, L = diagnostics(sol, r); dF = diff(F; dims = 1)
        Fin, _ = diagnostics(sol, test[:, :, 2:nrec, :]); Fout, _ = diagnostics(sol, p1[:, :, 2:nrec, :])     # 참 입력에서 한 걸음의 에너지
        x = reshape(test[:, :, 2:end, :], 64, 64, :)[:, :, 1:8:end]; gx = fno_apply(p, x, cons)                     # 대칭 점검용 상태 100개
        odd = mean(rel_each(-fno_apply(p, -x, cons), gx))
        sh(a) = circshift(a, (5, 17, 0)); equi = mean(rel_each(fno_apply(p, sh(x), cons), sh(gx)))
        inc = abs.(mean(p1 .- test[:, :, 1:end-1, :]; dims = (1, 2)))
        @printf("\n--- %s: %s (training %.0f s) ---\n", tag, NAMES[tag], models[tag].seconds)
        @printf("  one-step error (true input):   0->2.5: %.4f | 2.5->5: %.4f | 47.5->50: %.4f | mean of steps 2..20: %.4f\n", mean(e1[1, :]), mean(e1[2, :]), mean(e1[20, :]), mean(e1[2:end, :]))
        @printf("  rollout from the true c(2.5):  t=5: %.4f | t=12.5: %.4f | t=25: %.4f | t=50: %.4f\n", mean(er[1, :]), mean(er[4, :]), mean(er[9, :]), mean(er[19, :]))
        @printf("  [1] mass: max |mean(t) - mean(2.5)| over 40 rollouts x 19 steps = %.2e; mean |mean of one-step increment| = %.2e\n", maximum(mass_drift(r)), mean(inc))
        @printf("  [2] free energy: increases along the rollouts %d / %d; one step from true states: F(G(c)) > F(c) in %d / %d; F(50) = %.3f (solver %.3f)\n",
                count(>(0), dF), length(dF), count(Fout .> Fin), length(Fin), mean(F[end, :]), mean(Fs[end, :]))
        @printf("  [3] growth law: L(25) = %.3f, L(50) = %.3f (solver %.3f, %.3f); slope on 5 <= t <= 50: %.3f (solver %.3f)\n",
                mean(L[10, :]), mean(L[end, :]), mean(Ls[10, :]), mean(Ls[end, :]), growth_slope(t1, vec(mean(L; dims = 2))), growth_slope(t1, vec(mean(Ls; dims = 2))))
        @printf("  bounds: max |c| = %.3f (solver %.3f);  symmetries: ||G(-c) + G(c)|| / ||G(c)|| = %.4f,  ||G(Sc) - S G(c)|| / ||G(c)|| = %.4f\n",
                maximum(abs, r), maximum(abs, test), odd, equi)
        for mm in (0.3, 0.4, 0.5)                                       # 분포 밖: 학습 범위를 벗어난 평균 조성
            o = ood[:, :, :, findall(m -> abs(m) == mm, D.m_ood)]
            ro = rollout(p, o[:, :, 2, :], nrec - 1, cons); eo = rel4(ro[:, :, 2:end, :], o[:, :, 3:end, :]); Fo, Lo = diagnostics(sol, ro)
            @printf("  OOD |m| = %.1f: one-step (steps 2..20) %.4f | rollout t=5 %.4f, t=25 %.4f, t=50 %.4f | mass drift %.1e | F increases %d / %d | L(50) %.2f / solver %.2f\n",
                    mm, mean(rel4(one_step(p, o, cons), o[:, :, 2:end, :])[2:end, :]), mean(eo[1, :]), mean(eo[9, :]), mean(eo[19, :]), maximum(mass_drift(ro)),
                    count(>(0), diff(Fo; dims = 1)), length(Fo) - size(Fo, 2), mean(Lo[end, :]), mean(char_length(sol, o[:, :, end, :])))
        end
    end
    xs = reshape(test[:, :, 2:end, 1:8], 64, 64, :)[:, :, 1:4:end]; gs = advance(sol, xs, 50)
    @printf("\nsolver, same symmetry checks: odd %.1e, translation %.1e\n", mean(rel_each(-advance(sol, -xs, 50), gs)),
            mean(rel_each(advance(sol, circshift(xs, (5, 17, 0)), 50), circshift(gs, (5, 17, 0)))))
    # 비용: 큰 한 걸음(Δt = 2.5)의 FNO 대 솔버 50걸음, 그리고 손익분기
    p32 = models[:cons_pf].p; x1 = D.test[:, :, 5, 1:1]; c1 = test[:, :, 5, 1:1]
    ms_fno = 1e3 * best_time(() -> fno2d(p32, x1; conserve = true); rep = 20); ms_sol = 1e3 * best_time(() -> advance(sol, c1, 50); rep = 20)
    once = D.seconds + models[:cons_pf].seconds
    @printf("one big step (Dt = 2.5): FNO %.2f ms, solver (50 steps) %.2f ms -> speed-up %.2fx; one-off cost %.0f s", ms_fno, ms_sol, ms_sol / ms_fno, once)
    println(ms_sol > ms_fno ? @sprintf("; break-even after %.0f trajectories of 19 big steps", once / (19 * (ms_sol - ms_fno) / 1e3)) : "; no break-even: the solver is faster per big step")
end

foreach(train_model, (:free, :cons, :cons_pf)); evaluate()
