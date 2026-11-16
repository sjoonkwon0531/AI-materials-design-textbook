# lab_8_7.jl — Lab 8.7: 3층 재진입 라인, 디스패칭 규칙 대 REINFORCE (labs/82_fab.py)
# 이산 사건 시뮬레이터(장비 3대: 증착·리소·식각, 공정 시간 로그정규 CV 0.25)를 Julia로 옮기고,
# Gymnasium FabEnv에 대응하는 환경 구조체(reset!/step!)와 선형 softmax 디스패처의 '시간 국소' REINFORCE를 그대로 옮겼다.
# 사건 큐는 DataStructures.jl의 BinaryMinHeap. 학습 4회 + 시험(30 시나리오 x 2000 h x 정책 13개 x 도착률 2개).
include("ch8_common.jl")
using DataStructures

const ROUTE = [(1, 0.9), (2, 1.0), (3, 0.6), (1, 0.6), (2, 1.5), (3, 1.0), (1, 1.1), (2, 0.7), (3, 0.8)]   # (장비, 평균 시간)
const RAW = sum(last.(ROUTE))                                   # 8.2 h
const LITHO = sum(p for (s, p) in ROUTE if s == 2)              # 3.2 h/lot: 병목
const KMAX, NF = 24, 7

mutable struct Lot
    id::Int
    arr::Float64
    due::Float64
    step::Int          # 끝낸 공정 단계 수 (0..9)
    qarr::Float64
    out::Float64       # 완료 시각 (NaN = 아직 라인 안)
end
mutable struct Fab
    rng::Xoshiro
    rate::Float64
    H::Float64
    df::Float64
    t::Float64
    ev::BinaryMinHeap{Tuple{Float64,Int,Int}}    # (시각, 일련번호, 종류: 0 = 도착, s = 장비 s 완료)
    cnt::Int
    q::Vector{Vector{Lot}}
    busy::Vector{Union{Nothing,Lot}}
    lots::Vector{Lot}
    done::Vector{Lot}
    area::Float64      # WIP의 시간 적분
    last::Float64
    wip::Int
    log::Vector{Tuple{Float64,Float64}}
end
function Fab(rate, horizon, seed; due_factor = 2.5)
    f = Fab(Xoshiro(seed), rate, horizon, due_factor, 0.0, BinaryMinHeap{Tuple{Float64,Int,Int}}(), 0,
            [Lot[] for _ in 1:3], Union{Nothing,Lot}[nothing, nothing, nothing], Lot[], Lot[], 0.0, 0.0, 0, [(0.0, 0.0)])
    push_ev!(f, randexp(f.rng) / rate, 0)
    return f
end
push_ev!(f, t, kind) = (f.cnt += 1; push!(f.ev, (t, f.cnt, kind)))
function advance!(f, t)
    f.area += f.wip * (t - f.last); f.last = t; f.t = t; push!(f.log, (t, f.area))
end
ptime(f, lot) = ROUTE[lot.step+1][2] * exp(-0.5 * 0.0606 + 0.246 * randn(f.rng))       # 평균 p, CV ~0.25
enqueue!(f, lot) = (lot.qarr = f.t; push!(f.q[ROUTE[lot.step+1][1]], lot))
function start!(f, s, k)
    lot = popat!(f.q[s], k); f.busy[s] = lot
    push_ev!(f, f.t + ptime(f, lot), s)
end
"쉬는 장비 앞에 2개 이상이 기다릴 때(진짜 선택)까지 진행한다. 혼자 기다리는 lot은 바로 시작. 반환: 장비 번호 또는 nothing"
function pending_decision!(f)
    while true
        for s in 1:3
            f.busy[s] === nothing && length(f.q[s]) == 1 && start!(f, s, 1)
        end
        for s in 1:3
            f.busy[s] === nothing && length(f.q[s]) >= 2 && return s
        end
        if isempty(f.ev) || first(f.ev)[1] > f.H
            advance!(f, f.H); return nothing
        end
        t, _, kind = pop!(f.ev); advance!(f, t)
        if kind == 0                                            # 도착
            lot = Lot(length(f.lots) + 1, t, t + f.df * RAW, 0, t, NaN)
            push!(f.lots, lot); f.wip += 1; enqueue!(f, lot)
            push_ev!(f, t + randexp(f.rng) / f.rate, 0)
        else                                                    # 장비 kind 완료
            lot = f.busy[kind]; f.busy[kind] = nothing; lot.step += 1
            if lot.step == length(ROUTE)
                lot.out = t; push!(f.done, lot); f.wip -= 1
            else
                enqueue!(f, lot)
            end
        end
    end
end
"장비 s 앞 lot들의 특징 (lot x 7): 단계, 이번 시간, 남은 일, 여유, 이 큐의 대기, 라인 체류, 다음 장비의 큐 길이"
function features(f, s)
    F = zeros(length(f.q[s]), NF)
    for (r, lot) in enumerate(f.q[s])
        k = lot.step; p = ROUTE[k+1][2]; rem = sum(last.(ROUTE[k+2:end]); init = 0.0)
        nxt = k + 1 < length(ROUTE) ? length(f.q[ROUTE[k+2][1]]) : 0
        F[r, :] = [k / 8, p, rem / RAW, (lot.due - f.t - p - rem) / RAW, (f.t - lot.qarr) / RAW, (f.t - lot.arr) / RAW, nxt / 5]
    end
    return F
end

const RULES = Dict(
    "FIFO" => F -> argmax(F[:, 5]), "FIFO (fab)" => F -> argmax(F[:, 6]), "SPT" => F -> argmin(F[:, 2]),
    "LBFS" => F -> argmax(F[:, 1]), "FBFS" => F -> argmin(F[:, 1]), "least slack" => F -> argmin(F[:, 4]), "random" => nothing)
const RULE_ORDER = ["FIFO", "FIFO (fab)", "SPT", "LBFS", "FBFS", "least slack", "random"]

function simulate(policy, rate, horizon, seed; warm = 0.0)
    f = Fab(rate, horizon, seed); rng = Xoshiro(seed + 12345)
    while (s = pending_decision!(f)) !== nothing
        F = features(f, s)
        start!(f, s, policy === nothing ? rand(rng, 1:size(F, 1)) : policy(F))
    end
    done = [l for l in f.done if l.arr >= warm]
    ct = [l.out - l.arr for l in done]
    stuck = [horizon - l.arr for l in f.lots if isnan(l.out)]
    return (ct = mean(ct), ct95 = quantile(ct, 0.95), thr = length(f.done) / horizon, wip = f.area / horizon,
            late = mean(l.out > l.due for l in done), stuck = length(stuck), oldest = isempty(stuck) ? 0.0 : maximum(stuck),
            littleCT = (f.area / horizon) / (length(f.lots) / horizon))
end

# ---------------------------------------------------------------- Gymnasium FabEnv에 대응하는 환경
"관측: (KMAX x NF) 특징(0 채움)과 마스크. 보상: -(직전 결정 이후 WIP 적분)/RAW. Little 법칙으로 평균 체류 시간 최소화와 같다"
mutable struct FabEnv
    rate::Float64
    H::Float64
    rng::Xoshiro
    fab::Fab
    s::Union{Nothing,Int}
    area0::Float64
end
FabEnv(; rate = 0.2875, horizon = 300.0, seed = 0) = FabEnv(rate, horizon, Xoshiro(seed), Fab(rate, horizon, 0), nothing, 0.0)
function obs(env)
    F = features(env.fab, env.s)[1:min(end, KMAX), :]; L = zeros(KMAX, NF); L[1:size(F, 1), :] = F
    return (lots = L, mask = [i <= size(F, 1) for i in 1:KMAX])
end
function reset!(env)
    env.s = nothing
    while env.s === nothing
        env.fab = Fab(env.rate, env.H, rand(env.rng, 0:2^31-1)); env.s = pending_decision!(env.fab)
    end
    env.area0 = env.fab.area
    return obs(env)
end
"반환: (관측, 보상, terminated, truncated)"
function step!(env, a)
    start!(env.fab, env.s, mod(a - 1, min(length(env.fab.q[env.s]), KMAX)) + 1)
    env.s = pending_decision!(env.fab)
    r = -(env.fab.area - env.area0) / RAW; env.area0 = env.fab.area
    done = env.s === nothing
    return (done ? (lots = zeros(KMAX, NF), mask = falses(KMAX)) : obs(env)), r, done, false
end

# ---------------------------------------------------------------- 시간 국소 REINFORCE (선형 softmax 디스패처)
"결정 시각 t의 '수익' = -[t, t + window]의 WIP 적분/RAW, 기저선 = 같은 시나리오(공통 난수)를 공유한 K개 실행의 평균"
function reinforce_window(MU, SD; iters = 120, K = 8, lr = 0.3, horizon = 500.0, rate = 0.2875, window = 40.0, seed = 0, theta0 = nothing)
    rng = Xoshiro(seed); theta = theta0 === nothing ? zeros(NF) : copy(theta0); hist = Float64[]
    for it in 1:iters
        scen = rand(rng, 0:10^6-1); runs = []
        for k in 0:K-1
            f = Fab(rate, horizon, scen); prng = Xoshiro(scen * 31 + k); dec = Tuple{Float64,Vector{Float64}}[]
            while (s = pending_decision!(f)) !== nothing
                Phi = (features(f, s) .- MU') ./ SD'; p = softmax_vec(Phi * theta)
                a = findfirst(cumsum(p) .>= rand(prng)); a === nothing && (a = length(p))
                push!(dec, (f.t, Phi[a, :] .- Phi' * p)); start!(f, s, a)            # ∇ log π = φ_a - E_π[φ]
            end
            push!(runs, (first.(f.log), last.(f.log), dec, f.area / horizon))
        end
        Gfun(T_, A_, t) = -(interp1(min(t + window, horizon), T_, A_) - interp1(t, T_, A_)) / RAW
        grad = zeros(NF); nd = 0
        for (T_, A_, dec, _) in runs
            isempty(dec) && continue
            for (t, sc) in dec
                b = mean(Gfun(T2, A2, t) for (T2, A2, _, _) in runs)
                grad .+= (Gfun(T_, A_, t) - b) .* sc; nd += 1
            end
        end
        theta .+= lr .* grad ./ max(nd, 1)
        push!(hist, -mean(r[4] for r in runs))
    end
    return theta, hist
end

# ---------------------------------------------------------------- 실행
env = FabEnv(; seed = 0); o = reset!(env); Rsum = 0.0; ndec = 0
while true
    _, r, d, _ = step!(env, 1); global Rsum += r; global ndec += 1
    d && break
end
@printf("FabEnv episode (always first lot): %d decisions, return %.2f\n", ndec, Rsum)
rate = 0.2875
@printf("route: %d steps, raw process time %.1f h; litho load %.1f h/lot; arrival rate %.4f lots/h -> litho utilization %.2f\n",
        length(ROUTE), RAW, LITHO, rate, rate * LITHO)
f = Fab(rate, 300, 777); FF = Matrix{Float64}[]                  # 특징 표준화: FIFO 실행 하나에서
while (s = pending_decision!(f)) !== nothing
    F = features(f, s); push!(FF, F); start!(f, s, argmax(F[:, 5]))
end
FF = reduce(vcat, FF); MU = vec(mean(FF; dims = 1)); SD = vec(std(FF; dims = 1, corrected = false)) .+ 1e-8
fnames = ["step", "p_now", "remaining", "slack", "wait_q", "time_in_fab", "next_queue"]
pols = Pair{String,Any}[k => RULES[k] for k in RULE_ORDER]
th0 = zeros(NF); th0[4] = -1.0                                    # = least slack (온도 1의 softmax)
"선형 softmax 정책에서 lot 하나를 뽑는 확률적 디스패처"
function stoch(theta, seed)
    rng = Xoshiro(seed)
    return F -> begin
        p = softmax_vec(((F .- MU') ./ SD') * theta)
        something(findfirst(cumsum(p) .>= rand(rng)), length(p))
    end
end
for (name, init, sd) in [("RL seed0", nothing, 0), ("RL seed1", nothing, 1), ("RL seed2", nothing, 2), ("RL from least-slack", th0, 0)]
    theta, hist = reinforce_window(MU, SD; seed = sd, theta0 = init)
    println("$name: theta = ", join(["$n $(@sprintf("%+.2f", t))" for (n, t) in zip(fnames, theta)], ", "),
            @sprintf(" | training mean WIP first10 %.2f last10 %.2f", -mean(hist[1:10]), -mean(hist[end-9:end])))
    th = copy(theta)
    push!(pols, "$name (greedy)" => (F -> argmax(((F .- MU') ./ SD') * th)))
    name == "RL seed0" && push!(pols, "$name (stochastic)" => stoch(th, 0))
end
push!(pols, "least slack (softmax T=1)" => stoch(th0, 0))
println("\ntest: 30 scenarios, horizon 2000 h, statistics over lots released after 200 h warm-up (mean +/- SE)")
for rr in (rate, 0.30)
    @printf("arrival rate %.4f (litho utilization %.2f):\n", rr, rr * LITHO)
    W = Dict{String,Vector{Float64}}()
    for (name, pol) in pols
        Rs = [simulate(pol, rr, 2000.0, 5000 + s; warm = 200.0) for s in 0:29]
        W[name] = [r.wip for r in Rs]
        @printf("  %-28s WIP %5.1f+/-%.1f | Little CT %6.1f h | CT(finished) %6.1f | p95 %6.1f | late %.2f | thr %.4f | left in fab %4.1f, oldest %5.0f h\n",
                name, mean(W[name]), std(W[name]; corrected = false) / sqrt(30), mean(r.littleCT for r in Rs), mean(r.ct for r in Rs),
                mean(r.ct95 for r in Rs), mean(r.late for r in Rs), mean(r.thr for r in Rs), mean(r.stuck for r in Rs), mean(r.oldest for r in Rs))
    end
    best_rule = argmin(k -> mean(W[k]), filter(!=("random"), RULE_ORDER))
    for (name, _) in pols
        startswith(name, "RL") || continue
        d = W[name] .- W[best_rule]
        @printf("  paired WIP difference %s - %s: %+.2f +/- %.2f\n", name, best_rule, mean(d), std(d; corrected = false) / sqrt(length(d)))
    end
end
