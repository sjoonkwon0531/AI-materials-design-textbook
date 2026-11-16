# ch6_common.jl — Chapter 6 Julia 포트의 공용 함수
# 데이터 읽기(Magpie 캐시의 CSV판), 화학식 파서, 화학계·계열, 전처리, k-NN 회귀·k-평균, 분할(KFold·GroupKFold·층화),
# 닫힌 형식 ridge, L2 로지스틱 회귀(Newton), 지표.
# 파이썬 판과 같은 약속: StandardScaler는 ddof = 0, 분산 0인 열은 스케일 1.
using CSV, DataFrames, Statistics, LinearAlgebra, Random, Printf, StatsBase, NearestNeighbors, Clustering

const LABS = normpath(joinpath(@__DIR__, "..", "labs"))

# ---------------------------------------------------------------- 화학식 파서
"화학식 문자열(괄호 포함, 소수 계수 허용) -> (원소 목록, 양). 원소는 처음 나온 순서."
function parse_formula(s::AbstractString)
    c = collect(s); pos = Ref(1)
    function addto!(e, a, x, v)
        k = findfirst(==(x), e)
        k === nothing ? (push!(e, x); push!(a, v)) : (a[k] += v)
        return nothing
    end
    function number!()
        j = pos[]
        while j <= length(c) && (isdigit(c[j]) || c[j] == '.')
            j += 1
        end
        v = j > pos[] ? parse(Float64, String(c[pos[]:j-1])) : 1.0
        pos[] = j
        return v
    end
    function group!()
        e = String[]; a = Float64[]
        while pos[] <= length(c) && c[pos[]] != ')'
            if c[pos[]] == '('
                pos[] += 1
                se, sa = group!()
                pos[] += 1                       # ')' 건너뛰기
                m = number!()
                for (x, v) in zip(se, sa)
                    addto!(e, a, x, v * m)
                end
            else
                j = pos[] + 1
                while j <= length(c) && islowercase(c[j])
                    j += 1
                end
                x = String(c[pos[]:j-1]); pos[] = j
                addto!(e, a, x, number!())
            end
        end
        return e, a
    end
    return group!()
end

"화학계 = 원소 기호를 정렬해 '-'로 이은 문자열 (pymatgen 판과 같다)"
chemsys(f::AbstractString) = join(sort(parse_formula(f)[1]), "-")

"환원 조성의 정규형: 원소 정렬 + 최소 정수비(정수 계수일 때). 고유 조성 수를 세는 데 쓴다."
function reduced_key(f::AbstractString)
    e, a = parse_formula(f)
    o = sortperm(e); e, a = e[o], a[o]
    if all(isinteger, a)
        g = reduce(gcd, Int.(a)); a = a ./ g
    end
    return join(("$(x)$(v)" for (x, v) in zip(e, a)), " ")
end

"Pauling 전기음성도 표(labs/bartel_electronegativities.csv; 비활성 기체 0)"
function load_en()
    en = Dict{String,Float64}()
    for line in eachline(joinpath(LABS, "bartel_electronegativities.csv"))
        k, v = split(strip(line), ","); en[k] = parse(Float64, v)
    end
    return en
end

const FAMILY = Dict("F" => "halide", "Cl" => "halide", "Br" => "halide", "I" => "halide", "O" => "oxide",
                    "S" => "chalcogenide", "Se" => "chalcogenide", "Te" => "chalcogenide",
                    "N" => "pnictide", "P" => "pnictide", "As" => "pnictide", "Sb" => "pnictide")

"가장 전기음성도가 큰 원소로 정한 화학 계열 (동률이면 먼저 나온 원소)"
function family(f::AbstractString, en)
    e = parse_formula(f)[1]
    best = e[1]
    for x in e[2:end]
        get(en, x, 0.0) > get(en, best, 0.0) && (best = x)
    end
    return get(FAMILY, best, "other")
end

# ---------------------------------------------------------------- 데이터
"Magpie 캐시(CSV판). 반환: df, X(n x p), y, 짧은 특징 이름"
function load_magpie(; nonmetal_only = false)
    df = CSV.read(joinpath(LABS, "magpie_expt_gap.csv"), DataFrame)
    nonmetal_only && (df = df[df.target .> 0, :])
    feats = [c for c in names(df) if startswith(c, "MagpieData")]
    X = Matrix{Float64}(df[:, feats]); y = Vector{Float64}(df.target)
    return df, X, y, replace.(feats, "MagpieData " => "")
end

"열마다 최빈값이 차지하는 비율"
mode_share(x) = maximum(values(countmap(x))) / length(x)

"상수 열과 (선택) 최빈값 비율 > thr인 준상수 열을 뺀 열 마스크"
keep_mask(X; thr = 1.0) = [std(c) > 0 && mode_share(c) <= thr for c in eachcol(X)]

# ---------------------------------------------------------------- 전처리·모델
"StandardScaler(ddof = 0): (mu, sd) 반환, 분산 0인 열의 sd는 1"
function scaler(X)
    mu = vec(mean(X; dims = 1)); sd = vec(std(X; dims = 1, corrected = false))
    sd[sd .== 0] .= 1.0
    return mu, sd
end
zscore_with(X, (mu, sd)) = (X .- mu') ./ sd'

"Ridge(sklearn과 같은 목적함수): 절편이 있으면 X, y를 중심화한 뒤 (Xc'Xc + aI) w = Xc'yc"
function ridge_fit(X, y, a; intercept = true)
    if intercept
        mx = vec(mean(X; dims = 1)); my = mean(y)
        Xc = X .- mx'; w = Symmetric(Xc' * Xc + a * I) \ (Xc' * (y .- my))
        return w, my - dot(mx, w)
    else
        return Symmetric(X' * X + a * I) \ (X' * y), 0.0
    end
end
ridge_predict((w, b), X) = X * w .+ b

"표준화 + ridge (학습 겹으로만 스케일러를 맞춘다)"
function ridge_pipe(Xtr, ytr, Xte, a)
    sc = scaler(Xtr)
    m = ridge_fit(zscore_with(Xtr, sc), ytr, a)
    return ridge_predict(m, zscore_with(Xte, sc))
end

sigmoid(z) = 1 / (1 + exp(-z))

"L2 로지스틱 회귀, sklearn 목적함수 0.5|w|^2 + C*sum(logloss), 절편은 벌점 없음. Newton 반복."
function logreg_fit(X, y; C = 1.0, iters = 100, tol = 1e-10)
    n, p = size(X); A = hcat(X, ones(n)); w = zeros(p + 1)
    R = Diagonal(vcat(fill(1 / C, p), 0.0))
    for _ in 1:iters
        q = sigmoid.(A * w)
        g = A' * (q .- y) .+ R * w
        H = A' * (A .* (q .* (1 .- q))) + R
        step = Symmetric(H) \ g
        w .-= step
        maximum(abs, step) < tol && break
    end
    return w
end
logreg_proba(w, X) = sigmoid.(hcat(X, ones(size(X, 1))) * w)

"k-NN 회귀 (weights = 'uniform' 또는 'distance'), 유클리드 거리, 전수 탐색"
function knn_predict(Xtr, ytr, Xte, k; weights = :uniform)
    tree = BruteTree(permutedims(Xtr)); idx, dst = knn(tree, permutedims(Xte), k, true)
    p = similar(ytr, size(Xte, 1))
    for i in eachindex(p)
        if weights == :distance
            d = dst[i]
            if any(iszero, d)                      # sklearn: 거리 0인 이웃이 있으면 그것들만 평균
                p[i] = mean(ytr[idx[i][d .== 0]])
            else
                w = 1 ./ d; p[i] = sum(w .* ytr[idx[i]]) / sum(w)
            end
        else
            p[i] = mean(ytr[idx[i]])
        end
    end
    return p
end

"k-평균 n_init번 중 비용이 가장 작은 결과의 라벨"
function kmeans_best(Z, k; n_init = 10, seed = 0)
    rng = MersenneTwister(seed); best = nothing
    for _ in 1:n_init
        r = kmeans(permutedims(Z), k; maxiter = 300, rng = rng)
        (best === nothing || r.totalcost < best.totalcost) && (best = r)
    end
    return assignments(best)
end

# ---------------------------------------------------------------- 분할
"KFold(셔플): 앞의 n % k개 겹이 한 개씩 더 크다 (sklearn과 같은 크기 규칙, 순열은 Julia RNG)"
function kfold(n, k; rng = MersenneTwister(0), shuffle = true)
    idx = shuffle ? randperm(rng, n) : collect(1:n)
    sizes = fill(n ÷ k, k); sizes[1:(n % k)] .+= 1
    folds = Tuple{Vector{Int},Vector{Int}}[]; s = 0
    for f in 1:k
        te = sort(idx[s+1:s+sizes[f]]); s += sizes[f]
        push!(folds, (setdiff(1:n, te), te))
    end
    return folds
end

"GroupKFold: 큰 그룹부터 가장 가벼운 겹에 배정 (sklearn의 탐욕 규칙; 동률 그룹의 순서는 numpy와 다를 수 있다)"
function group_kfold(groups, k)
    ug = sort(unique(groups)); gid = Dict(g => i for (i, g) in enumerate(ug))
    gi = [gid[g] for g in groups]
    cnt = zeros(Int, length(ug)); for g in gi; cnt[g] += 1; end
    order = reverse(sortperm(cnt))                 # np.argsort(cnt)[::-1]의 안정 정렬판
    load = zeros(k); g2f = zeros(Int, length(ug))
    for g in order
        f = argmin(load); load[f] += cnt[g]; g2f[g] = f
    end
    fold = g2f[gi]
    return [(findall(!=(f), fold), findall(==(f), fold)) for f in 1:k]
end

"한 그룹씩 빼는 분할 (LeaveOneGroupOut; 그룹 값 순서대로)"
leave_one_group_out(groups) = [(findall(!=(g), groups), findall(==(g), groups)) for g in sort(unique(groups))]

"층화 k겹: 클래스마다 섞어 겹에 돌려 가며 나눈다"
function stratified_kfold(y, k; rng = MersenneTwister(0))
    fold = zeros(Int, length(y))
    for c in unique(y)
        idx = shuffle(rng, findall(==(c), y))
        for (t, i) in enumerate(idx)
            fold[i] = mod1(t, k)
        end
    end
    return [(findall(!=(f), fold), findall(==(f), fold)) for f in 1:k]
end

"그룹 단위 홀드아웃(GroupShuffleSplit 1회): 그룹을 섞어 ceil(test_size * 그룹 수)개를 시험으로"
function group_shuffle_split(groups; test_size = 0.2, rng = MersenneTwister(42))
    ug = shuffle(rng, sort(unique(groups)))
    tg = Set(ug[1:ceil(Int, test_size * length(ug))])
    te = findall(g -> g in tg, groups)
    return setdiff(1:length(groups), te), te
end

"교차검증 예측: fitpred(Xtr, ytr, Xte) -> 예측"
function cv_predict(fitpred, X, y, folds)
    p = zeros(length(y))
    for (tr, te) in folds
        p[te] = fitpred(X[tr, :], y[tr], X[te, :])
    end
    return p
end

# ---------------------------------------------------------------- 지표
mae(a, b) = mean(abs.(a .- b))
rmse(a, b) = sqrt(mean((a .- b) .^ 2))

"ROC-AUC (Mann-Whitney U; 동률은 평균 순위)"
function roc_auc(y, s)
    r = tiedrank(s); n1 = sum(y .== 1); n0 = length(y) - n1
    return (sum(r[y .== 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
end
