# lab_7_13.jl — Lab 7.13: 구면조화함수, Wigner D, 패리티를 수치로 확인하기 (labs/73b_sh_wigner.py)
# l ≤ 2는 so3.jl의 손으로 쓴 Y_l과 D^l로, l = 3–6은 PythonCall로 e3nn을 부르는 패턴으로 보인다(아래 끝).
# Part E의 SOAP 유사 계수는 l ≤ 2를 성분으로, 모든 l의 노름을 르장드르 덧셈 정리로 구한다.
include("so3.jl")
using Printf, Statistics, NPZ, FastGaussQuadrature

function main()
    rng = Xoshiro(0)
    println("=== Part A: Y_l(R r) = D^l(R) Y_l(r), 1000 random vectors, 10 random rotations ===")
    for T in (Float32, Float64)
        r = randn(rng, T, 3, 1000); errs = zeros(3)
        for l in 0:2, _ in 1:10
            R = T.(rand_rotation(rng)); D = wigner_D(l, R)
            e = maximum(maximum(abs.(sh(l, R * v) - D * sh(l, v))) for v in eachcol(r))
            errs[l+1] = max(errs[l+1], e)
        end
        println(rpad(string(T), 8), " max error by l=0..2: ", join([@sprintf("%.1e", x) for x in errs], " "))
    end

    println("=== Part B: D^l is orthogonal; a direct sum is block diagonal ===")
    R = rand_rotation(rng)
    for l in 1:2
        D = wigner_D(l, R)
        @printf("  l=%d: size %s, ||D'D - I||_max = %.1e, det = %+.6f\n", l, size(D), maximum(abs.(D' * D - I)), det(D))
    end
    Dsum = cat(wigner_D(0, R), wigner_D(0, R), wigner_D(1, R), wigner_D(2, R); dims = (1, 2))   # 2x0e + 1x1o + 1x2e
    blocks = [1:1, 2:2, 3:5, 6:10]; off = copy(Dsum)
    for b in blocks
        off[b, b] .= 0
    end
    @printf("  '2x0e+1x1o+1x2e' dim=%d; max |off-block| = %.1e;  sum_{l<=6}(2l+1) = %d = (l_max+1)^2\n",
            size(Dsum, 1), maximum(abs.(off)), sum(2l + 1 for l in 0:6))

    println("=== Part C: parity ===")
    r = randn(rng, 3, 500)
    for l in 0:2
        @printf("  l=%d: max|Y(-r) - (-1)^l Y(r)| = %.1e\n", l, maximum(maximum(abs.(sh(l, -v) - (-1)^l * sh(l, v))) for v in eachcol(r)))
    end
    Iv = -Matrix(1.0I, 3, 3)
    println("  D(inversion) for 1o: ", diag(irrep_D(1, -1, Iv)), "  for 1e: ", diag(irrep_D(1, 1, Iv)))
    a, b = randn(rng, 3), randn(rng, 3)
    for (name, M) in (("proper rotation", rand_rotation(rng)), ("improper (rotation x inversion)", -rand_rotation(rng)))
        lhs = cross(M * a, M * b)
        @printf("  cross product under %s: |err as 1o| = %.1e, |err as 1e| = %.1e\n", name,
                maximum(abs.(lhs - M * cross(a, b))), maximum(abs.(lhs - det(M) * M * cross(a, b))))
    end

    println("=== Part D: orthonormality ('integral' normalization Y/sqrt(4 pi)), Gauss-Legendre 40 x uniform phi 80 ===")
    x, w = gausslegendre(40); G = zeros(9, 9)
    for (xi, wi) in zip(x, w), k in 0:79
        ph = 2pi * k / 80; st = sqrt(1 - xi^2)
        y = vcat(sh(0, [st * cos(ph), st * sin(ph), xi]), sh(1, [st * cos(ph), st * sin(ph), xi]), sh(2, [st * cos(ph), st * sin(ph), xi])) ./ sqrt(4pi)
        G .+= wi * (2pi / 80) .* (y * y')
    end
    @printf("  Gram matrix 9 x 9 (l <= 2): max|G - I| = %.1e\n", maximum(abs.(G - I)))

    println("=== Part E: SOAP-like coefficients of ethanol atom C1 (frame 0; 'integral' normalization) ===")
    d = npzread(joinpath(@__DIR__, "..", "data", "ethanol_xtb_300K.npz"))
    X = permutedims(d["R"][1, :, :])                                        # 3 x 9
    rij = X[:, 2:end] .- X[:, 1]; dist = vec(sqrt.(sum(abs2, rij; dims = 1)))
    fc = @. 0.5 * (cos(pi * dist / 4) + 1) * (dist < 4)
    U = rij ./ dist'
    for l in 0:3
        Rr = rand_rotation(rng)
        normL = sqrt(legendre_power(l, U, fc)) / sqrt(4pi)                   # 모든 l: 덧셈 정리
        if l <= 2
            c = sum(fc[j] .* sh(l, rij[:, j]) for j in eachindex(fc)) ./ sqrt(4pi)
            cR = sum(fc[j] .* sh(l, Rr * rij[:, j]) for j in eachindex(fc)) ./ sqrt(4pi)
            @printf("  l=%d: |c_l| = %.6f (Legendre %.6f), ||c_l(R) - D c_l||_max = %.1e, ||c_l(R) - c_l||_max = %.3f\n",
                    l, norm(c), normL, maximum(abs.(cR - wigner_D(l, Rr) * c)), maximum(abs.(cR - c)))
        else
            @printf("  l=%d: |c_l| = %.6f (Legendre addition theorem; components need l = 3 harmonics)\n", l, normL)
        end
    end
end

main()

# ---------------------------------------------------------------- l > 2와 e3nn 대조: PythonCall 패턴
# using PythonCall
# torch = pyimport("torch"); o3 = pyimport("e3nn.o3"); torch.set_default_dtype(torch.float64)
# Rt = o3.rand_matrix(); R = pyconvert(Matrix{Float64}, Rt.numpy())
# r = randn(1000, 3)                                                      # 파이썬 쪽 배치 (N, 3)
# for l in 0:6
#     Y  = pyconvert(Matrix{Float64}, o3.spherical_harmonics(l, torch.tensor(r), normalize = true, normalization = "component").numpy())
#     YR = pyconvert(Matrix{Float64}, o3.spherical_harmonics(l, torch.tensor(r * R'), normalize = true, normalization = "component").numpy())
#     D  = pyconvert(Matrix{Float64}, o3.Irrep(l, (-1)^l).D_from_matrix(Rt).numpy())
#     println(l, "  ", maximum(abs.(YR - Y * D')))
# end
# l = 2 성분 대조: e3nn의 Y₂(x, y, z)는 여기 sh(2, [z, x, y])와 같다(축 이름만 다르다).
