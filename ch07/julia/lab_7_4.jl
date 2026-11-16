# lab_7_4.jl — Lab 7.4: Magpie 밴드갭의 double descent (labs/71_double_descent.py)
# 무작위로 고정한 ReLU 첫 층 + 최소 노름 최소제곱(유사역행렬) 또는 ridge로 맞춘 출력층. 순수 선형대수라
# 패키지 없이 옮겼다. 쌍대(커널) 형식 K = Φ Φᵀ (n x n)를 쓰므로 N = 30000에서도 n = 600 크기의 문제만 푼다.
include("ch7_common.jl")

X, y, _ = load_magpie()
tr, te = read_idx("fold0_train.csv"), read_idx("fold0_test.csv")
sub = read_idx("dd_sub600.csv")                 # 파이썬이 뽑은 600개 부분표본(같은 것을 쓰려면); 없으면 아래 줄
# sub = tr[randperm(Xoshiro(0), length(tr))[1:600]]
P = fit_prep(X[sub, :])
Xs, Xt = P(X[sub, :]), P(X[te, :])             # n x p (행 = 관측): 커널 형식은 이 배치가 편하다
ys, yt = y[sub], y[te]
n, p = size(Xs); ym = mean(ys)
Ns = [30, 100, 300, 450, 550, 590, 600, 610, 650, 750, 1000, 2000, 5000, 10000, 30000]
@printf("train n=%d, test %d, p=%d standardized Magpie features\n", n, length(te), p)
@printf("%7s %10s %18s %9s %21s\n", "N feat", "train RMSE", "test RMSE minnorm", "||w||", "test RMSE ridge 10")
for N in Ns
    trs, tes, ws, rid = Float64[], Float64[], Float64[], Float64[]
    for s in 0:2
        W = randn(Xoshiro(100 + s), p, N) ./ sqrt(p)
        Phi, Phit = max.(Xs * W, 0), max.(Xt * W, 0)
        K, Kt = Phi * Phi', Phit * Phi'
        alpha = pinv(Symmetric(K); rtol = 1e-12) * (ys .- ym)     # w = Φᵀ K⁺ y = Φ⁺ y (최소 노름 해)
        push!(trs, rmse(K * alpha .+ ym, ys))
        push!(tes, rmse(max.(Kt * alpha .+ ym, 0), yt))
        push!(ws, sqrt(max(dot(alpha, K * alpha), 0)))
        al2 = (K + 10.0 * I) \ (ys .- ym)                          # ridge λ = 10, 같은 쌍대 형식
        push!(rid, rmse(max.(Kt * al2 .+ ym, 0), yt))
    end
    @printf("%7d %10.3f %18.3f %9.2e %21.3f\n", N, median(trs), median(tes), median(ws), median(rid))
end
println("(medians over 3 random first layers; predictions clipped at 0)")
@printf("baseline (predict train mean): test RMSE %.3f\n", rmse(fill(ym, length(yt)), yt))
