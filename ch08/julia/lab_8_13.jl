# lab_8_13.jl — Lab 8.13: 행동 복제는 왜 자기 궤적에서 무너지는가, BC 대 DAgger (labs/84_dagger.py)
# 전문가: Lab 8.12의 명목 REINFORCE 정책(시드 0). 학습자: 외삽하지 못하는 회귀 나무(잎 최소 5개).
# scikit-learn DecisionTreeRegressor 대신 작은 CART(분산 감소, 중간값 문턱)를 직접 썼다. DecisionTree.jl의 회귀 나무로 바꿔도 된다.
include("ch8_common.jl")
include("furnace.jl")

# ---------------------------------------------------------------- 전문가
const THFILE = joinpath(@__DIR__, "84_theta_nominal_s0.txt")
const TH_EXPERT = isfile(THFILE) ? parse.(Float64, split(read(THFILE, String))) :
                  [2.036, 1.264, 0.397, 0.968, 0.899, -0.742, -0.234, 0.039]       # 파이썬 판 출력의 θ (소수 셋째 자리)
expert = linpolicy(TH_EXPERT)
const PL = (NOM.h, NOM.G, NOM.d)
const DN = 0.0005                                         # 시연·데이터 수집 때의 잡음 (평가는 0.003)

# ---------------------------------------------------------------- 작은 CART 회귀 나무
struct TNode
    feat::Int          # 0이면 잎
    thr::Float64
    left::Int
    right::Int
    value::Float64
end
"X: 특징 x 표본, y: 표본. 분산(제곱 오차 합) 감소가 가장 큰 축 정렬 분할을 재귀적으로 고른다"
function fit_tree(X, y; min_leaf = 5)
    nodes = TNode[]
    function grow(idx)
        k = length(nodes) + 1; push!(nodes, TNode(0, 0.0, 0, 0, mean(y[idx])))
        n = length(idx); (n < 2min_leaf || var(y[idx]; corrected = false) < 1e-14) && return k
        best = (0.0, 0, 0.0)                               # (감소량, 특징, 문턱)
        sse0 = sum(abs2, y[idx] .- mean(y[idx]))
        for f in 1:size(X, 1)
            o = idx[sortperm(X[f, idx])]; xs, ys = X[f, o], y[o]
            cs, cs2 = cumsum(ys), cumsum(ys .^ 2)
            for i in min_leaf:n-min_leaf
                xs[i] == xs[i+1] && continue
                sl = cs2[i] - cs[i]^2 / i; sr = (cs2[n] - cs2[i]) - (cs[n] - cs[i])^2 / (n - i)
                (sse0 - sl - sr > best[1]) && (best = (sse0 - sl - sr, f, (xs[i] + xs[i+1]) / 2))
            end
        end
        best[2] == 0 && return k
        f, thr = best[2], best[3]
        l = grow(idx[X[f, idx] .<= thr]); r = grow(idx[X[f, idx] .> thr])
        nodes[k] = TNode(f, thr, l, r, nodes[k].value)
        return k
    end
    grow(collect(1:length(y)))
    return nodes
end
function predict_tree(nodes, x)
    k = 1
    while nodes[k].feat != 0
        k = x[nodes[k].feat] <= nodes[k].thr ? nodes[k].left : nodes[k].right
    end
    return nodes[k].value
end
treepolicy(nodes) = o -> [predict_tree(nodes, view(o, :, i)) for i in 1:size(o, 2)]

# ---------------------------------------------------------------- BC 대 DAgger
"한 라운드에 n_ep 에피소드. BC는 언제나 전문가의 궤적을, DAgger는 0라운드 뒤로 학습자의 궤적에 전문가가 붙인 레이블을 모은다"
function run_dagger(seed; n_ep = 2, iters = 6, T = T_EP)
    rng = Xoshiro(seed); out = Dict("bc" => Float64[], "dagger" => Float64[], "n" => Int[])
    Obc, Ubc, Odg, Udg = Matrix{Float64}[], Vector{Float64}[], Matrix{Float64}[], Vector{Float64}[]
    pi_bc = pi_dg = nothing
    for k in 0:iters-1
        o = reshape(rollout(expert, PL..., n_ep, rng; T, record = true, noise = DN).obs, 7, :)
        push!(Obc, o); push!(Ubc, expert(o))
        if k == 0
            push!(Odg, o); push!(Udg, expert(o))
        else
            o2 = reshape(rollout(treepolicy(pi_dg), PL..., n_ep, rng; T, record = true, noise = DN).obs, 7, :)
            push!(Odg, o2); push!(Udg, expert(o2))                                  # 학습자가 간 상태에 전문가 레이블
        end
        pi_bc = fit_tree(reduce(hcat, Obc), reduce(vcat, Ubc)); pi_dg = fit_tree(reduce(hcat, Odg), reduce(vcat, Udg))
        push!(out["bc"], mean(rollout(treepolicy(pi_bc), PL..., 200, Xoshiro(10_000 + k); T).cost))
        push!(out["dagger"], mean(rollout(treepolicy(pi_dg), PL..., 200, Xoshiro(10_000 + k); T).cost))
        push!(out["n"], sum(length, Ubc))
    end
    # 분포 이동 진단: 학습자가 간 상태의 BC 학습 데이터까지 최근접 거리 (표준화)
    D = reduce(hcat, Obc); sd = vec(std(D; dims = 2, corrected = false)) .+ 1e-9
    nn(Q) = [minimum(sqrt.(vec(sum(abs2, (D .- q) ./ sd; dims = 1)))) for q in eachcol(Q)]
    r_bc = reshape(rollout(treepolicy(pi_bc), PL..., 50, Xoshiro(7); T, record = true).obs, 7, :)
    r_ex = reshape(rollout(expert, PL..., 50, Xoshiro(7); T, record = true).obs, 7, :)
    out["d"] = [median(nn(r_ex)), quantile(nn(r_ex), 0.9), median(nn(r_bc)), quantile(nn(r_bc), 0.9)]
    return out
end

@printf("expert on nominal plant: mean |e| %.1f K\n", 1e3 * mean(rollout(expert, PL..., 1000, Xoshiro(3)).cost))
R = [run_dagger(s) for s in 0:4]; n = R[1]["n"]
for key in ("bc", "dagger")
    A = 1e3 .* reduce(hcat, [r[key] for r in R])                                     # 라운드 x 시드
    println(rpad(key, 7), " ", join([@sprintf("n=%5d: %5.1f+/-%4.1f K", n[k], mean(A[k, :]), std(A[k, :]; corrected = false)) for k in eachindex(n)], " | "))
end
d = mean(reduce(hcat, [r["d"] for r in R]); dims = 2)
@printf("NN distance to BC data (standardised, median / p90): expert-visited %.3f / %.3f, BC-learner-visited %.3f / %.3f\n", d...)
for T in (30, 60, 120, 240)                                                          # 같은 예산(6 라운드 x 2 에피소드), 더 긴 유지
    rs = [run_dagger(s; T) for s in 0:4]
    eT = 1e3 * mean(rollout(expert, PL..., 500, Xoshiro(3); T).cost)
    bc, dg = [1e3 * r["bc"][end] for r in rs], [1e3 * r["dagger"][end] for r in rs]
    @printf("T=%3d: expert %5.1f K | BC %6.1f +/- %5.1f K | DAgger %5.1f +/- %4.1f K  (total cost per episode: BC %.1f, DAgger %.1f, expert %.1f K s)\n",
            T, eT, mean(bc), std(bc; corrected = false), mean(dg), std(dg; corrected = false), T * mean(bc), T * mean(dg), T * eT)
end
