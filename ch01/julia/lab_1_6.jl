# Lab 1.6 (Julia): Newton Fractal — 초기값 민감성의 시각화
# f(z) = z^3 - 1 에 복소 Newton 반복을 격자의 모든 점에서 수행.
# Julia에서는 벡터화 대신 점별(pointwise) 루프가 자연스럽고 빠르다(JIT 컴파일).
using Printf, Statistics

const roots = ComplexF64[1.0, exp(2im * pi / 3), exp(4im * pi / 3)]
const N, span, max_iter, tol = 1000, 2.0, 50, 1e-6

"""한 초기점 z에서 Newton 반복. (basin, iterations) 반환; basin은 1..3, 미수렴이면 0."""
function newton_point(z::ComplexF64)
    for k in 1:max_iter                       # Python의 k+1에 해당 (반복 횟수 = k)
        df = 3.0 * z^2
        step = abs(df) > 1e-12 ? (z^3 - 1.0) / df : zero(z)
        z -= step
        for ri in 1:3
            if abs(z - roots[ri]) < tol
                return ri, k
            end
        end
    end
    return 0, 0
end

function main()
    x = range(-span, span; length = N)
    # NumPy meshgrid(x, x): X[i,j] = x[j], Y[i,j] = x[i]
    basin = zeros(Int, N, N)
    itc   = zeros(Int, N, N)
    for j in 1:N, i in 1:N                    # column-major 순서로 순회
        basin[i, j], itc[i, j] = newton_point(complex(x[j], x[i]))
    end

    conv = basin .> 0
    @printf("converged fraction = %.4f\n", mean(conv))          # 1.0000
    @printf("mean iterations    = %.3f\n", mean(itc[conv]))     # 6.843
    println("basin fractions    = ", [round(mean(basin .== i), digits = 4) for i in 1:3])
    # -> [0.3528, 0.3236, 0.3236]
    @printf("max iterations     = %d\n", maximum(itc))          # 47
end

main()
