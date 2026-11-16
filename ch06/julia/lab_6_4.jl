# Lab 6.4 (Julia) — 랭크 결손 특징 행렬과 세 가지 교차검증
# 파이썬 labs/63_linreg_cv.py 대응. 비금속 2,151행, Magpie 특징(캐시를 CSV로 내보낸 labs/magpie_expt_gap.csv).
# Part A(조건수, 수치 랭크, QR 대 SVD 최소 노름 해, 영공간 방향)는 결정론적이라 숫자가 그대로 일치해야 한다
# (특이값 1e-14 근처의 작은 값과 QR 해의 거대한 노름은 LAPACK 반올림에 따라 자릿수만 같다).
# Part B: 무작위 KFold의 순열과 k-평균 초기화는 Julia RNG라서 통계적으로 일치하고, 그룹 5겹은
# sklearn GroupKFold의 탐욕 규칙을 그대로 옮겼지만 같은 크기 그룹의 순서(numpy argsort의 동률 처리)가
# 달라 겹 구성이 조금 다르다. 숫자는 소수 둘째 자리 근처까지 비슷하다.
using LinearAlgebra, Printf, Random, Statistics, NearestNeighbors, Clustering
include("ch6_common.jl")

function main()
    t0 = time()
    d, X, y, names = load_magpie(; nonmetal_only = true)
    keep = keep_mask(X); X = X[:, keep]; names = names[keep]
    @printf("non-metal subset: n = %d, p = %d (after dropping constant columns)\n", length(y), size(X, 2))
    @printf("gap: mean %.3f eV, std %.3f eV, min %.2f, max %.2f\n", mean(y), std(y; corrected = false), minimum(y), maximum(y))

    # ---------- Part A
    Xs = zscore_with(X, scaler(X)); A = hcat(Xs, ones(length(y)))
    s = svdvals(A)
    println("\n=== Part A: OLS on standardized Magpie features ===")
    @printf("cond(A) = %.3e   cond(A^T A) = %.3e\n", s[1] / s[end], (s[1] / s[end])^2)
    println("singular values: largest ", @sprintf("%.1f", s[1]), ", 10 smallest: ", join([@sprintf("%.2e", v) for v in s[end-9:end]], ", "))
    @printf("number of sigma_i < 1e-6 * sigma_1: %d\n", count(<(1e-6 * s[1]), s))
    @printf("numerical rank of A = %d of %d columns\n", rank(A), size(A, 2))
    F = lu(Symmetric(A' * A) |> Matrix; check = false)
    println(issuccess(F) ? "normal eq.: LU finished (the solution is numerically meaningless)" : "normal eq.: LU reports a singular matrix")
    Fq = qr(A); Rq = Fq.R
    w_qr = UpperTriangular(Rq) \ (Matrix(Fq.Q)' * y)
    @printf("QR: min |R_ii| = %.2e\n", minimum(abs.(diag(Rq))))
    Fs = svd(A); tol = eps() * maximum(size(A)) * Fs.S[1]          # numpy lstsq(rcond=None)와 같은 문턱
    w_svd = Fs.V * (map(x -> x > tol ? 1 / x : 0.0, Fs.S) .* (Fs.U' * y))
    for (nm, w) in (("QR (no pivoting)", w_qr), ("SVD/lstsq (min-norm)", w_svd))
        @printf("%-21s: train RMSE = %.4f eV   ||w|| = %.3e   max|w| = %.3e\n", nm, rmse(A * w, y), norm(w), maximum(abs.(w)))
    end
    v0 = Fs.V[:, end]; w2 = w_svd + 10.0 * v0
    @printf("w_SVD + 10*v_null: ||w|| = %.3f, max |A w2 - A w_SVD| = %.2e\n", norm(w2), maximum(abs.(A * w2 - A * w_svd)))
    ix(n) = findfirst(==(n), names)
    @printf("check exact collinearity: max |range - (max - min)| for Number = %.1e\n",
            maximum(abs.(X[:, ix("range Number")] - (X[:, ix("maximum Number")] - X[:, ix("minimum Number")]))))

    # ---------- Part B
    cs = chemsys.(d.formula)
    cm_ = countmap(cs); cnt = collect(values(cm_))
    big = argmax(cm_)
    println("\n=== Part B: how the split changes the error ===")
    @printf("chemical systems: %d distinct; %d systems hold >1 entry, covering %d of %d rows; largest: %s (%d)\n",
            length(cm_), count(>(1), cnt), sum(cnt[cnt .> 1]), length(y), big, cm_[big])
    qc = [mode_share(c) > 0.99 for c in eachcol(X)]
    println("quasi-constant features (>99% identical): ", sum(qc), " -> ", join(names[qc], ", "))
    Xf = X[:, .!qc]
    models = [("ridge, $(size(X, 2)) feat.", X, (a, b, c) -> ridge_pipe(a, b, c, 1.0)),
              ("ridge, $(size(Xf, 2)) feat.", Xf, (a, b, c) -> ridge_pipe(a, b, c, 1.0)),
              ("1-NN, $(size(Xf, 2)) feat.", Xf, (a, b, c) -> (sc = scaler(a); knn_predict(zscore_with(a, sc), b, zscore_with(c, sc), 1))),
              ("5-NN, $(size(Xf, 2)) feat.", Xf, (a, b, c) -> (sc = scaler(a); knn_predict(zscore_with(a, sc), b, zscore_with(c, sc), 5; weights = :distance)))]
    clus = kmeans_best(zscore_with(Xf, scaler(Xf)), 8)
    println("leave-cluster-out: k-means (k=8, 10 inits) on standardized features; cluster sizes (sorted): ",
            sort(collect(values(countmap(clus))); rev = true))
    schemes = [("random 5-fold", kfold(length(y), 5; rng = MersenneTwister(0))),
               ("group 5-fold (chem. system)", group_kfold(cs, 5)),
               ("leave-cluster-out (k=8)", leave_one_group_out(clus))]
    @printf("%-18s%s\n", "model", join([lpad(k, 30) for (k, _) in schemes]))
    worst = nothing
    for (mn, XX, fp) in models
        row = Float64[]
        for (sn, folds) in schemes
            p = cv_predict(fp, XX, y, folds)
            push!(row, mae(p, y))
            if startswith(mn, "ridge, 130") && startswith(sn, "random")
                e = abs.(p - y); k = argmax(e)
                worst = (d.formula[k], e[k], median(e), mean(e[setdiff(1:end, k)]))
            end
        end
        @printf("%-18s%s\n", mn, join([@sprintf("%30.3f", v) for v in row]))
    end
    @printf("ridge/130, random 5-fold: worst point %s |error| = %.3e eV; median AE = %.3f; MAE without it = %.3f\n",
            worst[1], worst[2], worst[3], worst[4])
    @printf("(baseline: predict the training mean -> MAE = %.3f eV)\n", mae(y, fill(mean(y), length(y))))
    println("median distance (standardized feature space) from a test point to its nearest training point:")
    for (sn, folds) in schemes
        out = zeros(length(y))
        for (tr, te) in folds
            sc = scaler(Xf[tr, :])
            tree = BruteTree(permutedims(zscore_with(Xf[tr, :], sc)))
            _, dst = knn(tree, permutedims(zscore_with(Xf[te, :], sc)), 1)
            out[te] = first.(dst)
        end
        @printf("   %-30s: %.3f\n", sn, median(out))
    end
    @printf("time %.1f s\n", time() - t0)
end

main()
