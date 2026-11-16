# Lab 1.8 (Julia): 같은 함수 f(x) = x^2 sin(x)를 4가지로 미분하여 비교한다.
# 필요 패키지: ] add Symbolics ForwardDiff
using Printf
using Symbolics      # 기호 미분 (SymPy 대응)
using ForwardDiff    # 자동미분 (forward mode, dual number; §1.8의 a + bε)

# === 방법 1: 손으로 구한 해석적 미분 ===
f_analytical_derivative(x) = 2x * sin(x) + x^2 * cos(x)

# === 방법 2: 기호 미분 (Symbolics.jl) ===
@variables xs
df_sym = expand_derivatives(Differential(xs)(xs^2 * sin(xs)))
f_symbolic = build_function(df_sym, xs; expression = Val{false})   # 컴파일된 Julia 함수

# === 방법 3: 수치 미분 (중심차분) ===
f(x) = x^2 * sin(x)
f_numerical_derivative(x, h = 1e-5) = (f(x + h) - f(x - h)) / (2h)

# === 방법 4: 자동미분 (ForwardDiff) ===
# 입력이 Float64이면 Dual{Float64}로 계산된다. 1.0f0(Float32)을 넣으면 FP32 정밀도로 떨어진다.
f_ad_derivative(x) = ForwardDiff.derivative(f, x)

# === 비교 ===
x0 = 1.0
exact = 2 * 1.0 * sin(1.0) + 1.0^2 * cos(1.0)

results = [
    "Analytical (by hand)" => f_analytical_derivative(x0),
    "Symbolic (Symbolics)" => f_symbolic(x0),
    "Numerical (h=1e-5)"   => f_numerical_derivative(x0, 1e-5),
    "Numerical (h=1e-10)"  => f_numerical_derivative(x0, 1e-10),
    "Automatic (ForwardDiff)" => f_ad_derivative(x0),
]

@printf("%-25s | %20s | %10s\n", "Method", "Value", "|Error|")
println("-"^65)
for (name, value) in results
    @printf("%-25s | %20.16f | %10.2e\n", name, value, abs(value - exact))
end
