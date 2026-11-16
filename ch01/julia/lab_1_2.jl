# Lab 1.2 (Julia): Floating-point precision in action
# Goal: 4가지 정밀도에서 동일 계산을 비교 (Python Lab 1.2의 Julia 대응)
using Printf

# 1. 머신 엡실론 직접 측정
"""1.0 + eps > 1.0이 성립하는 가장 작은 eps를 찾는다 (T = Float64, Float32, Float16)."""
function find_machine_epsilon(::Type{T}) where {T<:AbstractFloat}
    eps_ = one(T)
    while one(T) + eps_ / T(2) > one(T)
        eps_ = eps_ / T(2)
    end
    return eps_
end

println("--- Machine epsilon (measured) ---")
@printf("FP64: %.4e\n", find_machine_epsilon(Float64))
@printf("FP32: %.4e\n", find_machine_epsilon(Float32))
@printf("FP16: %.4e\n", find_machine_epsilon(Float16))   # Float16은 Base에 내장
# BF16: 산술은 BFloat16s.jl 패키지가 필요하다. 여기서는 형식 정의(가수 7+1 bit)로부터 직접 계산.
# (BFloat16s.jl이 있으면: using BFloat16s; eps(BFloat16) 와 같은 값)
bf16_eps = 2.0^-7
@printf("BF16: %.4e\n", bf16_eps)

# 2. Catastrophic cancellation 체험
println("\n--- Catastrophic cancellation ---")
for x_val in (1e2, 1e4, 1e6, 1e8, 1e10)
    x32 = Float32(x_val)
    x64 = Float64(x_val)
    naive32 = sqrt(x32^2 + 1) - x32          # 정수 1과의 연산은 Float32를 유지
    naive64 = sqrt(x64^2 + 1) - x64
    robust = 1.0 / (sqrt(x64^2 + 1) + x64)
    @printf("x=%.0e: naive(FP32)=%.4e, naive(FP64)=%.4e, robust(FP64)=%.4e\n",
            x_val, naive32, naive64, robust)
end

# 3. 누적 합 (associativity 위반)
println("\n--- Sum is not associative in FP ---")
small_values = fill(1.0f-8, 10^7)                  # Float32 배열
total_a = 1.0f0 + sum(small_values)                # Base.sum도 pairwise summation
total_b = sum(small_values) + 1.0f0                # 두 피연산자만 교환(결과 동일)
total_c = foldl(+, small_values; init = 1.0f0)     # 하나씩 순차 누적
@printf("Vectorized sum:    %.10f\n", total_a)
@printf("Commuted order:    %.10f\n", total_b)
@printf("Sequential sum:    %.10f\n", total_c)
# 결과: 누적 순서에 따라 결과가 다름!
