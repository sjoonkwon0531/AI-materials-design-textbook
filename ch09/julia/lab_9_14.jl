# lab_9_14.jl — Lab 9.14: 앙상블은 무엇을 놓치고 잔차는 무엇을 잡는가 (labs/94d_uq.py)
# lab_9_13.jl이 저장한 사전학습 모형 4개와 자료를 읽는다. 옮긴 것: (0) 잔차 연산자의 점검, (1) 앙상블 오차·퍼짐·포함률,
# (2) 라벨 없는 PDE 잔차와 오차의 순위 상관, (3) 분할 등각 예측의 오차 노름 상계, (4) 솔버로 되돌리기, (5) 손익분기.
# 줄인 포트: 장 전체에 대한 등각 띠 둘과 같은 정확도의 거친 솔버 탐색은 뺐다.
include("family.jl")

load(name) = deserialize(joinpath(CH9_OUT, "94_$(name).jls"))
const MEMBERS = [load("tfno_member$(k)").p for k in 0:3]
const TRAIN_S = mean(load("tfno_member$(k)").seconds for k in 0:3)

"앙상블 평균의 (u, u_t): 평균 예측의 접선은 접선의 평균이다"
function ens_jvp(u0, coef, t)
    r = [tfno_jvp(p, u0, coef, t) for p in MEMBERS]
    return mean(first.(r)), mean(last.(r))
end

function main()
    # (0) 정확한 해 exp(-ν k² t) sin k(x - ct)에서 잔차가 0인가 (Float64)
    x = (0:127) .* (2pi / 128); nu0, c0, k0, t = 0.1, 0.7, 3, 0.5
    u = reshape(exp(-nu0 * k0^2 * t) .* sin.(k0 .* (x .- c0 * t)), :, 1)
    ut = reshape(exp(-nu0 * k0^2 * t) .* (-nu0 * k0^2 .* sin.(k0 .* (x .- c0 * t)) .- c0 * k0 .* cos.(k0 .* (x .- c0 * t))), :, 1)
    @printf("(0) residual operator on the exact solution: rms r = %.1e\n", sqrt(mean(abs2, ut .- rhs(u, reshape([nu0, c0, 0.0, 0.0], 4, 1)))))

    sets = [("cal", "pre_cal"), ("ID", "pre_test"), ("low-nu", "lownu_test"), ("high-k", "highk_test"), ("reaction", "react_test")]
    S = Dict{String,Any}()
    println("\n(1)(2) ensemble of 4: error, spread, residual (300 trajectories each, block t = 0.1..1.0)")
    println("  set        err(member0)  err(ens)   mean sd    cover 1.96sd   rho_S(sd,err)  rho_S(res,err)   rms res")
    for (n, f) in sets
        d = load(f); tru = d.u[:, T_IDX, :]
        preds = stack([predict(p, d, T_IDX) for p in MEMBERS]; dims = 4)      # (N, 10, n, 4)
        mu = dropdims(mean(preds; dims = 4); dims = 4); sd = dropdims(std(preds; dims = 4); dims = 4)
        err = vec(sqrt.(mean(abs2, mu .- tru; dims = (1, 2))))                 # 궤적마다의 RMS 오차
        rel = rel_each(mu, tru); rel0 = rel_each(preds[:, :, :, 1], tru)
        spread = vec(sqrt.(mean(abs2, sd; dims = (1, 2))))                     # 궤적마다의 RMS 앙상블 표준편차
        res = residual_norms(ens_jvp, d)                                       # RMS PDE 잔차 (라벨 없음)
        cov = mean(abs.(tru .- mu) ./ sd .< 1.96)
        S[n] = (; err, rel, spread, res)
        @printf("  %-9s   %6.2f%%    %6.2f%%   %.4f     %5.1f%%          %5.2f          %5.2f        %.3f\n",
                n, 100 * mean(rel0), 100 * mean(rel), mean(sd), 100 * cov, spearman(spread, err), spearman(res, err), mean(res))
    end
    test = ["ID", "low-nu", "high-k", "reaction"]
    pool = (; (k => reduce(vcat, [getfield(S[n], k) for n in test]) for k in (:err, :rel, :spread, :res))...)
    bad = pool.rel .> 0.05
    @printf("  pooled (1200): Spearman(sd, err) = %.2f, Spearman(res, err) = %.2f; Pearson of logs: sd %.2f, res %.2f\n",
            spearman(pool.spread, pool.err), spearman(pool.res, pool.err), cor(log.(pool.spread), log.(pool.err)), cor(log.(pool.res), log.(pool.err)))
    @printf("  detecting 'relative error > 5%%' (%d of 1200): AUROC sd %.3f, residual %.3f\n", count(bad), auroc(pool.spread, bad), auroc(pool.res, bad))

    # (3) 분할 등각 예측: ID 보정 집합 300개, 명목 90%
    q_sd = conformal_q(S["cal"].err ./ S["cal"].spread); q_res = conformal_q(S["cal"].err ./ S["cal"].res)
    @printf("\n(3) split conformal (calibration: 300 ID trajectories, nominal 90%%): ||e|| <= q ||sd||: q = %.2f;  ||e|| <= q ||r||: q = %.3f\n", q_sd, q_res)
    for n in test
        s = S[n]
        @printf("  %-9s  bound from spread: coverage %5.1f%% (bound/error, median %4.1fx)   bound from residual: coverage %5.1f%% (%4.1fx)\n", n,
                100 * mean(s.err .<= q_sd .* s.spread), median(q_sd .* s.spread ./ s.err), 100 * mean(s.err .<= q_res .* s.res), median(q_res .* s.res ./ s.err))
    end

    # (4) 가장 의심스러운 궤적을 솔버로 되돌린다 (다시 푼 것은 오차 0으로 센다)
    println("\n(4) solver fallback on the pooled 1200 trajectories (mean relative error of what remains)")
    println("  fallback fraction     random    ensemble sd   PDE residual   oracle")
    rng = Xoshiro(0)
    for f in (0.0, 0.1, 0.25, 0.5)
        nfb = floor(Int, f * 1200)
        row = [100 * sum(pool.rel[sortperm(sc)[1:1200-nfb]]) / 1200 for sc in (rand(rng, 1200), pool.spread, pool.res, pool.rel)]
        @printf("        %4.2f           %5.2f%%      %5.2f%%       %5.2f%%      %5.2f%%\n", f, row[1], row[2], row[3], row[4])
    end

    # (5) 분할상환의 손익분기: N* = (자료 생성 + 학습) / (솔버 - 대리 모형)
    d = load("pre_test"); u0, coef, t1 = d.u[:, 1, :], d.coef, ones(Float32, 300)
    c_inf = best_time(() -> tfno(MEMBERS[1], u0, coef, t1); rep = 5) / 300
    c_ens = best_time(() -> mean(tfno(p, u0, coef, t1) for p in MEMBERS); rep = 5) / 300
    c_res = best_time(() -> pde_residual(MEMBERS[1], u0, coef, t1); rep = 5) / 300
    u0s = sample_ic(Xoshiro(3), 300, 256); cfs = sample_coeffs(Xoshiro(33), 300, :pre)
    c_sol = best_time(() -> etdrk4(u0s, cfs; n_save = 1); rep = 3) / 300
    c_data = 2000 * c_sol
    @printf("\n(5) cost per query u0 -> u(T), ms per trajectory: FNO member 0: %.3f | ensemble of 4: %.3f | member 0 + residual: %.3f | ETDRK4 (256, dt 5e-3): %.3f\n",
            1e3 * c_inf, 1e3 * c_ens, 1e3 * c_res, 1e3 * c_sol)
    for (who, ci, ct) in (("single FNO", c_inf, c_data + TRAIN_S), ("ensemble of 4", c_ens, c_data + 4 * TRAIN_S))
        @printf("  break-even vs reference solver: %-13s speed-up %6.1fx, N* = %.0f queries\n", who, c_sol / ci, c_sol > ci ? ct / (c_sol - ci) : Inf)
    end
end
main()
