# lab_8_6.jl — Lab 8.6: 장난감 반응망 위의 역합성, 탐욕·무작위·MCTS(UCT)·정확한 DP (labs/82_mcts.py)
# 패키지 없이 옮겼다(내보낸 반응망을 읽을 때만 JSON3). 분자 번호는 1부터. 반응 표를 조회할 때마다 분자를 기록해
# '단일 단계 역합성 모형 호출 수'(분자별 캐시)를 센다. AND-OR 트리의 Bellman 식 V(m) = max_r y_r Π V(p)가 정확한 답이다.
# 반응망은 Julia 난수로 새로 만들거나(통계적 비교), 파이썬이 만든 100개를 JSON으로 읽는다(DP 값이 그대로 일치).
include("ch8_common.jl")
using JSON3

const LEVELS = [30, 25, 25, 20, 15, 10, 1]          # 복잡도 단계별 분자 수; 단계 0 = 구매 가능 후보

mutable struct Net
    lev::Vector{Int}
    buyable::BitVector
    rxns::Dict{Int,Vector{Tuple{Float64,Vector{Int}}}}   # 생성물 -> [(수율, 전구체들)]
    target::Int
    seen::Set{Int}                                     # 반응 모형에 보낸 분자들
end
"반응 표 조회(기록됨)"
rx(net, m) = (push!(net.seen, m); get(net.rxns, m, Tuple{Float64,Vector{Int}}[]))

function make_network(seed)
    rng = Xoshiro(seed)
    lev = reduce(vcat, [fill(L, n) for (L, n) in zip(0:length(LEVELS)-1, LEVELS)])
    buyable = falses(length(lev)); buyable[lev .== 0] .= rand(rng, count(==(0), lev)) .< 0.7
    rxns = Dict{Int,Vector{Tuple{Float64,Vector{Int}}}}()
    for mol in eachindex(lev)
        L = lev[mol]; L == 0 && continue
        if L <= 2 && rand(rng) < 0.15
            rxns[mol] = Tuple{Float64,Vector{Int}}[]; continue      # 알려진 절단이 없음
        end
        out = Tuple{Float64,Vector{Int}}[]
        for _ in 1:rand(rng, 1:4)
            npre = rand(rng) < 0.4 ? 1 : 2
            pre = [rand(rng, findall(==(max(0, L - rand(rng, 1:2))), lev)) for _ in 1:npre]
            push!(out, (0.2 + 0.75 * rand(rng), pre))
        end
        rxns[mol] = out
    end
    return Net(lev, buyable, rxns, length(lev), Set{Int}())
end
"파이썬이 만든 반응망(JSON; 0-based)을 읽는다"
function load_networks(path)
    nets = Net[]
    for d in JSON3.read(read(path, String))
        rx_ = Dict{Int,Vector{Tuple{Float64,Vector{Int}}}}()
        for (k, v) in pairs(d[:rxns])
            rx_[parse(Int, String(k)) + 1] = [(Float64(r[1]), Int.(collect(r[2])) .+ 1) for r in v]
        end
        push!(nets, Net(Int.(collect(d[:lev])), BitVector(Bool.(collect(d[:buyable]))), rx_, length(d[:lev]), Set{Int}()))
    end
    return nets
end

"정확한 최적: V(m) = 1 (구매 가능), 아니면 max_r y_r Π V(p). 반환 (값, 펼친 분자 수)"
function dp_best(net)
    memo = Dict{Int,Float64}()
    function V(m)
        haskey(memo, m) && return memo[m]
        net.buyable[m] && return (memo[m] = 1.0)
        best = 0.0
        for (y, pre) in rx(net, m)
            best = max(best, y * prod(V(p) for p in pre))
        end
        return (memo[m] = best)
    end
    return V(net.target), length(memo)
end

"부분 경로를 끝까지: 스택 꼭대기의 미해결 분자에 반응을 고르고 전구체를 쌓는다. 막다른 길은 0"
function rollout_value(net, stack, rng, depth; maxdepth = 12, policy = :random)
    val = 1.0; stack = copy(stack)
    while !isempty(stack)
        m = pop!(stack); net.buyable[m] && continue
        rs = rx(net, m)
        (isempty(rs) || depth >= maxdepth) && return 0.0
        y, pre = policy == :random ? rs[rand(rng, 1:length(rs))] : rs[argmax(first.(rs))]
        val *= y; append!(stack, pre); depth += 1
    end
    return val
end
greedy_route(net) = rollout_value(net, [net.target], nothing, 0; policy = :greedy)

mutable struct Node
    stack::Vector{Int}
    val::Float64
    depth::Int
    children::Dict{Int,Node}
    N::Int
    W::Float64
    untried::Vector{Int}
    terminal::Bool
end
function Node(stack, val, depth, net)
    stack = filter(m -> !net.buyable[m], stack)                  # 구매 가능한 분자는 걷어 낸다
    terminal = isempty(stack) || isempty(rx(net, stack[end])) || depth >= 12
    untried = terminal ? Int[] : collect(1:length(rx(net, stack[end])))
    return Node(stack, val, depth, Dict{Int,Node}(), 0, 0.0, untried, terminal)
end
function mcts(net, n_sim; c = 0.4, seed = 0)
    rng = Xoshiro(seed); empty!(net.seen)
    root = Node([net.target], 1.0, 0, net); best = 0.0
    for _ in 1:n_sim
        node = root; path = [root]
        while !node.terminal && isempty(node.untried) && !isempty(node.children)     # 1) 선택(UCT)
            par = node
            node = argmax(ch -> ch.W / ch.N + c * sqrt(log(par.N) / ch.N), collect(values(par.children)))
            push!(path, node)
        end
        if !node.terminal && !isempty(node.untried)                                   # 2) 확장
            a = popat!(node.untried, rand(rng, 1:length(node.untried)))
            y, pre = rx(net, node.stack[end])[a]
            child = Node([node.stack[1:end-1]; pre], node.val * y, node.depth + 1, net)
            node.children[a] = child; node = child; push!(path, node)
        end
        v = node.terminal ? (isempty(node.stack) ? node.val : 0.0) :                  # 3) 모의 실행
            node.val * rollout_value(net, node.stack, rng, node.depth)
        best = max(best, v)
        for nd in path                                                                # 4) 역전파
            nd.N += 1; nd.W += v
        end
    end
    return best, length(net.seen)
end
function random_search(net, B, seed)
    rng = Xoshiro(seed); empty!(net.seen)
    v = maximum(rollout_value(net, [net.target], rng, 0) for _ in 1:B)
    return v, length(net.seen)
end

# ---------------------------------------------------------------- 실행
const JSON_PATH = joinpath(CH8_DATA, "82_mcts_networks.json")
nets = Tuple{Net,Float64,Int}[]
if isfile(JSON_PATH)                                               # 파이썬과 같은 100개
    for n in load_networks(JSON_PATH)
        v, e = dp_best(n); v > 0 && push!(nets, (n, v, e))
    end
else                                                               # Julia 난수로 새로 만든 망
    for s in 0:399
        n = make_network(s); v, e = dp_best(n)
        v > 0 && push!(nets, (n, v, e))
        length(nets) == 100 && break
    end
end
opt = [v for (_, v, _) in nets]; nexp = [e for (_, _, e) in nets]
@printf("networks: %d solvable; molecules %d, levels %d\n", length(nets), sum(LEVELS), length(LEVELS))
@printf("exact DP optimum (product of yields): mean %.3f, median %.3f; molecules DP had to expand: mean %.1f\n",
        mean(opt), median(opt), mean(nexp))
g = [greedy_route(n) for (n, _, _) in nets]
@printf("greedy (highest-yield step each time): success %.2f, value/optimum %.3f, optimal %.2f\n",
        mean(g .> 0), mean(g ./ opt), mean(isapprox.(g, opt)))
for B in (10, 30, 100, 300, 1000), (name, fn) in [("random rollouts", (n, i) -> random_search(n, B, i)),
                                                  ("MCTS (UCT c=0.4)", (n, i) -> mcts(n, B; c = 0.4, seed = i))]
    out = [fn(n, i - 1) for (i, (n, _, _)) in enumerate(nets)]
    vb = first.(out); q = last.(out)
    @printf("  B=%5d %-17s: success %.2f, value/optimum %.3f +/- %.3f, optimal %.2f, queried %.1f\n", B, name,
            mean(vb .> 0), mean(vb ./ opt), std(vb ./ opt; corrected = false) / sqrt(length(opt)), mean(isapprox.(vb, opt)), mean(q))
end
d = [mcts(n, 300; seed = i - 1)[1] / v - random_search(n, 300, i - 1)[1] / v for (i, (n, v, _)) in enumerate(nets)]
@printf("  paired difference MCTS - random at B=300: %.3f +/- %.3f (SE); MCTS better on %.2f, worse on %.2f\n",
        mean(d), std(d; corrected = false) / sqrt(length(d)), mean(d .> 1e-12), mean(d .< -1e-12))
