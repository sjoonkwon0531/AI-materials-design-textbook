# lab_7_20.jl — Lab 7.20: 데이터 병렬의 핵심, 기울기 평균 (labs/76_ddp.py의 Julia 대응; Distributed.jl판)
# PyTorch DDP는 프로세스마다 같은 모형을 두고, 역전파가 끝날 때 기울기를 all-reduce로 평균한 뒤 각자 같은 갱신을 한다.
# 여기서는 같은 수학을 Distributed.jl로 보인다: 작업자 2개가 전역 배치의 절반씩에서 국소 기울기를 계산하고,
# 주 프로세스가 평균해 갱신한다(매개변수 서버 꼴). 손실이 배치 평균이고 두 절반의 크기가 같으면
# 평균 기울기 = 전역 배치 기울기이므로, 단일 프로세스 학습과 반올림 오차 안에서 같은 매개변수에 도달한다.
# 진짜 all-reduce(MPI_Allreduce) 판은 lab_7_20_mpi.jl에 있다. 둘 다 작은 예제이고 실행해 보지는 못했다.
using Distributed
nprocs() == 1 && addprocs(2; exeflags = "--threads=1")
@everywhere begin
    include(joinpath($(@__DIR__), "ch7_common.jl"))
    include(joinpath($(@__DIR__), "flux_tools.jl"))
    BLAS.set_num_threads(1)
    const H = 128
    make_model() = mlp_chain(Xoshiro(0), [149, H, H, 1])                   # 모든 프로세스에서 같은 초기 가중치
    loss(m, x, y) = mean(abs.(vec(m(x)) .- y))
    "평탄화한 매개변수 θ에서 국소 미니배치의 기울기(평탄화)"
    function local_grad(θ, x, y)
        _, re = Flux.destructure(make_model())
        g = only(Zygote.gradient(t -> loss(re(t), x, y), θ))
        return g
    end
end

function load()
    X, y, _ = load_features149()
    P = fit_prep(X; thr = nothing, clipv = 6.0)                           # 파이썬과 같이 전체로 맞춘 표준화(이 Lab의 목적은 학습 역학)
    n = (length(y) ÷ 256) * 256
    return feats(P, X[1:n, :]), Float32.(y[1:n])
end

"단일 프로세스, 전역 배치 256, 에폭마다 같은 순열"
function single(X, y; epochs = 4, B = 256)
    θ, re = Flux.destructure(make_model())
    opt = Optimisers.setup(TorchAdamW(1e-3; lambda = 1e-2), θ)
    for ep in 0:epochs-1
        perm = randperm(Xoshiro(ep), length(y))
        for s in 1:B:length(y)
            idx = perm[s:s+B-1]
            g = only(Zygote.gradient(t -> loss(re(t), X[:, idx], y[idx]), θ))
            opt, θ = Optimisers.update!(opt, θ, g)
        end
    end
    return θ
end

"데이터 병렬: 전역 배치를 작업자 수로 나눠(DistributedSampler처럼 번갈아) 국소 기울기 → 평균 → 한 번 갱신"
function data_parallel(X, y; epochs = 4, B = 256)
    ws = workers(); W = length(ws)
    θ, _ = Flux.destructure(make_model())
    opt = Optimisers.setup(TorchAdamW(1e-3; lambda = 1e-2), θ)
    first_step = nothing
    for ep in 0:epochs-1
        perm = randperm(Xoshiro(ep), length(y))
        for s in 1:B:length(y)
            idx = perm[s:s+B-1]
            parts = [idx[r:W:end] for r in 1:W]                             # 순위 r의 몫
            futs = [remotecall(local_grad, ws[r], θ, X[:, parts[r]], y[parts[r]]) for r in 1:W]
            gs = fetch.(futs)
            g = sum(gs) ./ W                                                # all-reduce(합) / 순위 수 = 평균
            if first_step === nothing
                re = Flux.destructure(make_model())[2]                     # 모형 생성(push!)은 미분 밖에서
                gfull = only(Zygote.gradient(t -> loss(re(t), X[:, idx], y[idx]), θ))
                first_step = (norm.(gs), maximum(abs.(g .- gfull)))
            end
            opt, θ = Optimisers.update!(opt, θ, g)
        end
    end
    return θ, first_step
end

function main()
    X, y = load()
    @printf("training rows %d, global batch 256 = %d workers x %d\n", length(y), nworkers(), 256 ÷ nworkers())
    t1 = @elapsed θs = single(X, y)
    t2 = @elapsed (θd, fs) = data_parallel(X, y)
    @printf("local gradient norms (first batch): %s;  max |mean of local grads - full-batch grad| = %.1e\n", round.(fs[1]; digits = 4), fs[2])
    @printf("after training: max |theta_DP - theta_single| = %.1e (max |theta| = %.2f)\n", maximum(abs.(θd .- θs)), maximum(abs.(θs)))
    @printf("wall: single %.2f s | data parallel (%d workers, incl. communication) %.2f s\n", t1, nworkers(), t2)
end

main()
