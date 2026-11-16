# lab_7_20_mpi.jl — Lab 7.20의 MPI.jl 판: 순위마다 모형 사본, 기울기 Allreduce 평균, 각자 같은 갱신 (PyTorch DDP의 구조)
# 실행: mpiexecjl -n 2 julia --project lab_7_20_mpi.jl   (MPI.jl의 mpiexecjl 감싸개; 실행해 보지는 못했다)
using MPI
include("ch7_common.jl")
include("flux_tools.jl")

MPI.Init()
const comm = MPI.COMM_WORLD
const rank, world = MPI.Comm_rank(comm), MPI.Comm_size(comm)
BLAS.set_num_threads(1)

X, y, _ = load_features149()
P = fit_prep(X; thr = nothing, clipv = 6.0)
n = (length(y) ÷ 256) * 256
X, y = feats(P, X[1:n, :]), Float32.(y[1:n])

θ, re = Flux.destructure(mlp_chain(Xoshiro(rank), [149, 128, 128, 1]))   # 일부러 순위마다 다른 초기화
MPI.Bcast!(θ, comm; root = 0)                                                 # DDP처럼 순위 0의 가중치를 방송
opt = Optimisers.setup(TorchAdamW(1e-3; lambda = 1e-2), θ)
loss(t, x, yb) = mean(abs.(vec(re(t)(x)) .- yb))
B = 256
for ep in 0:3
    perm = randperm(Xoshiro(ep), length(y))                              # 모든 순위가 같은 순열(같은 시드)
    for s in 1:B:length(y)
        idx = perm[s:s+B-1][rank+1:world:end]                            # DistributedSampler: 순위 몫을 번갈아
        g = only(Zygote.gradient(t -> loss(t, X[:, idx], y[idx]), θ))
        MPI.Allreduce!(g, +, comm)                                       # 합
        g ./= world                                                      # 평균 = 전역 배치 기울기
        global opt, θ = Optimisers.update!(opt, θ, g)
    end
end
θ0 = copy(θ); MPI.Bcast!(θ0, comm; root = 0)
d = MPI.Allreduce(maximum(abs.(θ .- θ0)), max, comm)
rank == 0 && @printf("world %d: after 4 epochs, max |theta_rank - theta_rank0| = %.1e\n", world, d)
MPI.Finalize()
