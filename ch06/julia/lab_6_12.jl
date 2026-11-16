# Lab 6.12 (Julia) — 잔차를 쌓는 부스팅, 그리고 w*를 손으로 확인하기
# 파이썬 labs/65_boost.py 대응.
#  (A) 제곱 손실 그래디언트 부스팅을 직접 구현(약한 학습기 = DecisionTree.jl의 깊이 2 회귀 트리).
#      장난감 자료가 Julia RNG로 만들어지므로 오차 곡선은 통계적으로 일치한다(최소 시험 MSE ~0.07, M ~50).
#  (B) XGBoost 그루터기의 분기·잎 값·이득을 Julia에서 정확 탐색(exact greedy)으로 다시 찾고 손 계산과
#      비교한다. 결정론적이라 숫자가 그대로 일치해야 한다(mean NpValence < 1.7217, 이득 2055.093).
#      XGBoost.jl의 트리와도 대조한다(XGBoost.trees의 필드 이름은 2.x 판 기준).
using Printf, Random, Statistics
include("ch6_common.jl")
include("ensemble_tools.jl")

ftrue(x) = sin(2pi * x) + 0.5 * (x > 0.6)

"정확 탐색 그루터기: 제곱 손실, 기준값 base, L2 lambda. (이득, 특징, 문턱) 반환"
function best_stump(X, y, base, lam)
    g = base .- y; G = sum(g); H = length(y); best = (-Inf, 0, 0.0)
    for j in 1:size(X, 2)
        o = sortperm(X[:, j]); xs = X[o, j]; GL = cumsum(g[o])
        for i in 1:length(y)-1
            xs[i+1] > xs[i] || continue                          # 같은 값 사이에서는 나눌 수 없다
            gl, hl = GL[i], i
            gain = gl^2 / (hl + lam) + (G - gl)^2 / (H - hl + lam) - G^2 / (H + lam)
            gain > best[1] && (best = (gain, j, 0.5 * (xs[i] + xs[i+1])))
        end
    end
    return best
end

function main()
    # ---------------- Part A
    rng = MersenneTwister(1); n = 200
    x = sort(rand(rng, n)); y = ftrue.(x) .+ 0.25 .* randn(rng, n)
    xt = rand(rng, 2000); yt = ftrue.(xt) .+ 0.25 .* randn(rng, 2000)
    nu, M = 0.1, 500
    F = fill(mean(y), n); Ft = fill(mean(y), 2000); tr_c = Float64[]; te_c = Float64[]
    Xx, Xt = reshape(x, :, 1), reshape(xt, :, 1)
    for m in 1:M
        r = y .- F                                        # (1/2)(y - F)^2의 음의 그래디언트
        h = tree_fit(Xx, r; max_depth = 2)
        F .+= nu .* apply_tree(h, Xx); Ft .+= nu .* apply_tree(h, Xt)
        push!(tr_c, mean((y .- F) .^ 2)); push!(te_c, mean((yt .- Ft) .^ 2))
    end
    println("=== Part A: L2 gradient boosting, depth-2 trees, nu=0.1, n=200, noise sd=0.25 (var 0.0625) ===")
    for m in (1, 10, 50, 100, 200, 500)
        @printf("  M=%3d: train MSE = %.4f   test MSE (n=2000) = %.4f\n", m, tr_c[m], te_c[m])
    end
    @printf("  best test MSE %.4f at M=%d\n", minimum(te_c), argmin(te_c))

    # ---------------- Part B
    _, X, yv, names = load_magpie(); lam = 1.0; base = mean(yv)
    gain, j, thr = best_stump(X, yv, base, lam)
    g = base .- yv; L = X[:, j] .< thr
    GL, HL, GR, HR = sum(g[L]), sum(L), sum(g[.!L]), sum(.!L)
    println("=== Part B: one stump on expt_gap (n=4601), squared loss, lambda=1, eta=1, exact greedy in Julia ===")
    @printf("  best split found by exhaustive search: %s < %.4f\n", names[j], thr)
    @printf("  left : n=%4d  G=%9.2f  H=%6d  -G/(H+lam)=%+.5f\n", HL, GL, HL, -GL / (HL + lam))
    @printf("  right: n=%4d  G=%9.2f  H=%6d  -G/(H+lam)=%+.5f\n", HR, GR, HR, -GR / (HR + lam))
    @printf("  gain GL^2/(HL+lam)+GR^2/(HR+lam)-G^2/(H+lam) = %.3f\n", gain)
    # XGBoost.jl과 대조: 같은 설정의 트리 하나
    dm = DMatrix(Float32.(X); label = yv, feature_names = names)
    bst = xgboost(dm; num_round = 1, max_depth = 1, eta = 1.0, lambda = lam, base_score = base,
                  tree_method = "exact", min_child_weight = 0, objective = "reg:squarederror", watchlist = Dict())
    root = first(XGBoost.trees(bst; with_stats = true))
    @printf("  XGBoost.jl root: feature %s < %.4f, gain %.3f; leaves %+.5f, %+.5f\n", root.split, root.split_condition,
            root.gain, root.children[1].leaf, root.children[2].leaf)
end

main()
