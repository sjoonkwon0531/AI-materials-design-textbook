# Lab 4.3 (Julia) — DFT가 O(N^3)인 이유를 재 보기: 밀집 대칭행렬 대각화 시간 vs N
# 파이썬 판과 같은 크기·반복 수. 주의 두 가지:
#  (1) Julia는 첫 호출 때 JIT 컴파일을 하므로, 측정 전에 작은 행렬로 한 번 "예열(warm-up)"한다.
#  (2) BLAS 스레드를 1로 묶어 병렬화가 지수를 가리지 않게 한다(파이썬의 OPENBLAS_NUM_THREADS=1).
# 난수 행렬(Random.seed!(0))은 numpy와 다른 생성기라 같은 행렬이 아니지만, 측정 대상인
# 대각화 비용은 행렬 원소 값에 거의 무관하다. 절대 시간은 기계마다 다르고 지수만 의미가 있다.
using LinearAlgebra, Printf, Random, Statistics
BLAS.set_num_threads(1)
Random.seed!(0)

eigvals(Symmetric(randn(50, 50)))                    # JIT 예열 (측정에서 제외)

sizes = [200, 400, 800, 1200, 1600, 2400, 3200]
times = Float64[]
println("=== Diagonalization wall-time vs matrix size (LAPACK via eigvals(Symmetric)) ===")
println("   N      time(s)   reps")
for N in sizes
    A = randn(N, N); A = 0.5 .* (A .+ A')           # symmetric (Hermitian)
    reps = N <= 800 ? 5 : (N <= 1600 ? 3 : 1)
    t0 = time_ns()
    for _ in 1:reps
        eigvals(Symmetric(A))
    end
    dt = (time_ns() - t0) / 1e9 / reps
    push!(times, dt)
    @printf(" %5d   %8.4f   %d\n", N, dt, reps)
end

"log-log 1차 적합의 기울기 (np.polyfit(log N, log t, 1)[0]과 같다)"
slope(xs, ys) = ([ones(length(xs)) log.(xs)] \ log.(ys))[2]
p_full = slope(sizes, times)
mask = sizes .>= 800
p_tail = slope(sizes[mask], times[mask])
@printf("\nfull-range fit p = %.2f  |  asymptotic tail (N>=800) p = %.2f  (ideal=3)\n", p_full, p_tail)
println("local doubling exponents:")
for i in 2:length(sizes)
    r = sizes[i] / sizes[i-1]; f = times[i] / times[i-1]
    @printf("  %d->%d: time x%.2f  -> exponent %.2f\n", sizes[i-1], sizes[i], f, log(f) / log(r))
end
@printf("implied slowdown for 10x larger system (cubic): ~%dx\n", 10^3)
