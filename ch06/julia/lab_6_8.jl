# Lab 6.8 (Julia) — Magpie 공간의 t-SNE·UMAP: 무엇이 보존되고 무엇이 사라지는가
# 파이썬 labs/64_embed.py 대응. t-SNE는 TSne.jl(정확한 O(n^2) 구현; sklearn의 Barnes-Hut보다 느려
# 4,601점에서 한 번에 수 분 걸린다), UMAP은 UMAP.jl(0.1 계열 API: umap(X, d; n_neighbors, min_dist),
# X는 열 = 관측). trustworthiness, k-NN 겹침, Procrustes 불일치는 정의대로 직접 계산한다.
# 임베딩은 초기화와 최적화가 모두 난수에 의존하므로 모든 숫자가 통계적으로만 일치한다(PCA 행은 예외로
# 결정론적이지만 10-NN 정확도는 무작위 KFold에 의존한다). 비교할 것은 표의 순서와 자릿수다.
using LinearAlgebra, Printf, Random, Statistics, StatsBase, NearestNeighbors, TSne, UMAP
include("ch6_common.jl")

"각 점의 k-NN (자기 자신 제외), 행렬 k x n"
function knn_ids(E, k)
    tree = BruteTree(permutedims(E)); idx, _ = knn(tree, permutedims(E), k + 1, true)
    return reduce(hcat, [i[2:end] for i in idx])
end

"sklearn.manifold.trustworthiness와 같은 정의"
function trustworthiness(Z, E; k = 10)
    n = size(Z, 1); NE = knn_ids(E, k); t = 0.0; rk = zeros(Int, n)
    for i in 1:n
        dz = vec(sum((Z .- Z[i:i, :]) .^ 2; dims = 2)); dz[i] = Inf
        rk[sortperm(dz)] = 1:n                          # 입력 공간에서의 순위(가장 가까운 점 = 1)
        for j in view(NE, :, i)
            r = rk[j] - k; r > 0 && (t += r)
        end
    end
    return 1 - t * 2 / (n * k * (2n - 3k - 1))
end

knn_overlap(NH, E; k = 10) = mean(length(intersect(NH[:, i], c)) / k for (i, c) in enumerate(eachcol(knn_ids(E, k))))

"무작위 5겹에서 10-NN 다수결 분류 정확도"
function knn_cv_acc(E, lab; k = 10, rng = MersenneTwister(0))
    acc = Float64[]
    for (tr, te) in kfold(size(E, 1), 5; rng = rng)
        idx, _ = knn(BruteTree(permutedims(E[tr, :])), permutedims(E[te, :]), k, true)
        pred = [mode(lab[tr][i]) for i in idx]
        push!(acc, mean(pred .== lab[te]))
    end
    return mean(acc)
end

pdist(E) = [norm(E[i, :] - E[j, :]) for i in 1:size(E, 1) for j in i+1:size(E, 1)]

"scipy.spatial.procrustes의 불일치: 둘 다 중심화·Frobenius 정규화 후 1 - (sum sigma)^2"
function procrustes_disparity(A, B)
    A = A .- mean(A; dims = 1); A ./= norm(A); B = B .- mean(B; dims = 1); B ./= norm(B)
    return 1 - sum(svdvals(A' * B))^2
end

pca2(Z) = (Zc = Z .- mean(Z; dims = 1); Zc * svd(Zc).V[:, 1:2])
run_tsne(Z, perp) = tsne(Z, 2, 0, 1000, Float64(perp); progress = false)      # 입력: 행 = 관측
run_umap(Z, k; seed = 0) = (Random.seed!(seed); permutedims(umap(permutedims(Z), 2; n_neighbors = k, min_dist = 0.1)))

function main()
    t0 = time()
    d, X, y, _ = load_magpie(); X = X[:, keep_mask(X; thr = 0.99)]
    Z = zscore_with(X, scaler(X)); n, p = size(Z); metal = Int.(y .== 0)
    @printf("n = %d, p = %d\n", n, p)
    emb = Dict{String,Matrix{Float64}}()
    emb["PCA"] = pca2(Z)
    t = time(); emb["t-SNE (perp 30)"] = run_tsne(Z, 30); @printf("t-SNE time %.1f s\n", time() - t)
    t = time(); emb["UMAP (k=15)"] = run_umap(Z, 15); @printf("UMAP time %.1f s\n", time() - t)

    sub = randperm(MersenneTwister(0), n)[1:1000]; dh = pdist(Z[sub, :])
    NH = knn_ids(Z, 10)
    @printf("\n10-NN metal/non-metal accuracy in the %d-D standardized space (random 5-fold) = %.3f\n", p, knn_cv_acc(Z, metal))
    @printf("%-18s %10s %14s %10s %15s\n", "embedding", "trust(10)", "10-NN overlap", "10-NN acc", "Spearman(dist)")
    for k in ("PCA", "t-SNE (perp 30)", "UMAP (k=15)")
        E = emb[k]
        @printf("%-18s %10.3f %14.3f %10.3f %15.3f\n", k, trustworthiness(Z, E), knn_overlap(NH, E),
                knn_cv_acc(E, metal), corspearman(dh, pdist(E[sub, :])))
    end

    println("\n=== perplexity / n_neighbors sweep (t-SNE, UMAP; seed 0) ===")
    for perp in (5, 100)
        E = run_tsne(Z, perp)
        @printf("t-SNE perp %3d: trust = %.3f, 10-NN overlap = %.3f, Spearman(dist) = %.3f\n",
                perp, trustworthiness(Z, E), knn_overlap(NH, E), corspearman(dh, pdist(E[sub, :])))
    end
    for kk in (5, 50)
        E = run_umap(Z, kk)
        @printf("UMAP k = %3d     : trust = %.3f, 10-NN overlap = %.3f, Spearman(dist) = %.3f\n",
                kk, trustworthiness(Z, E), knn_overlap(NH, E), corspearman(dh, pdist(E[sub, :])))
    end

    println("\n=== seed sensitivity (UMAP k=15) ===")
    for s in (1, 2)
        E = run_umap(Z, 15; seed = s)
        @printf("seed 0 vs seed %d: Procrustes disparity = %.3f, trust = %.3f\n", s,
                procrustes_disparity(emb["UMAP (k=15)"], E), trustworthiness(Z, E))
    end

    println("\n=== synthetic test: three Gaussian blobs in 10-D ===")
    r = MersenneTwister(1)
    A = randn(r, 300, 10); B = 4.0 .* randn(r, 300, 10); C = randn(r, 300, 10)
    B[:, 1] .+= 30.0; C[:, 1] .+= 120.0
    S = vcat(A, B, C); lab = repeat(1:3; inner = 300)
    function summ(E)
        c = [vec(mean(E[lab .== i, :]; dims = 1)) for i in 1:3]
        rad = [mean(norm.(eachrow(E[lab .== i, :] .- c[i]'))) for i in 1:3]
        return rad[2] / rad[1], norm(c[3] - c[1]) / norm(c[2] - c[1])
    end
    for (k, E) in (("input (10-D)", S), ("t-SNE perp 30", run_tsne(S, 30)), ("UMAP k=15", run_umap(S, 15)), ("PCA", pca2(S)))
        rr, dr = summ(E)
        @printf("%-14s: radius(B)/radius(A) = %5.2f   dist(A,C)/dist(A,B) = %5.2f\n", k, rr, dr)
    end
    @printf("time %.1f s\n", time() - t0)
end

main()
