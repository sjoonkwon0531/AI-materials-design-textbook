# Lab 2.7 — 1D heat equation u_t = u_xx on [0,1], u(0)=u(1)=0, by the Method of Lines
# (Julia port of lab2_7.py; alpha = 1, N = 50 intervals, hat IC u0 = 1 - |2x - 1|, t_final = 0.2)
# Reference "semi-discrete exact" solution: exp(M t) u0 (dense matrix exponential, LinearAlgebra).
using LinearAlgebra, SparseArrays, Printf

lap(N) = (m = N - 1; spdiagm(-1 => ones(m - 1), 0 => -2 * ones(m), 1 => ones(m - 1)) * N^2)  # tridiag(1,-2,1)/h^2
lam_k(N, k) = 4 * N^2 * sin(k * pi / (2N))^2                    # |eigenvalue| k of lap(N)

# (1) smallest eigenmode, N = 50
N = 50
@printf("(1) N=50: lambda1 = %.3f  vs pi^2 = %.3f   (eigvals check %.4f)\n",
        lam_k(N, 1), pi^2, -maximum(eigvals(Symmetric(Matrix(lap(N))))))
# (2) eigenvalue error / h^2 -> pi^4/12 (2nd order)
println("(2) (pi^2 - lambda1)/h^2:")
for n in (10, 20, 40, 80, 160)
    @printf("    N=%4d  %.3f\n", n, (pi^2 - lam_k(n, 1)) * n^2)
end
@printf("    limit pi^4/12 = %.3f\n", pi^4 / 12)

# (3) time stepping on N = 50, hat IC, T = 0.2   (lambda = dt/h^2)
h = 1 / N; x = range(0, 1, length=N + 1); xi = x[2:end-1]; m = N - 1; T = 0.2
u0 = 1 .- abs.(2 .* xi .- 1)
Ah = Matrix(lap(N)) * h^2                                       # tridiag(1,-2,1)
uref = exp(Matrix(lap(N)) * T) * u0                             # e^{M T} u0
function march(lam, scheme)
    dt = lam * h^2; n = round(Int, T / dt); u = copy(u0)
    P = if scheme == :be
        inv(I - lam * Ah)                                       # backward Euler
    elseif scheme == :cn
        (I - 0.5 * lam * Ah) \ (I + 0.5 * lam * Ah)              # Crank–Nicolson
    else
        nothing
    end
    for _ in 1:n
        u = scheme == :ftcs ? u + lam * (Ah * u) : P * u
    end
    return n, u
end
for (label, lam, sch) in (("FTCS 0.40", 0.40, :ftcs), ("FTCS 0.52", 0.52, :ftcs), ("BE   10", 10.0, :be), ("CN   10", 10.0, :cn))
    n, u = march(lam, sch)
    @printf("(3) %-9s steps=%4d  max|u(0.2)|=%.4g  max|u - e^{Mt}u0|=%.2e\n", label, n, maximum(abs.(u)), maximum(abs.(u - uref)))
end
@printf("    reference max|e^{Mt}u0| = %.4f\n", maximum(abs.(uref)))

# (4) stiffness ratio |lambda_max|/|lambda_min| grows like (2/pi)^2 N^2
for n in (10, 20, 40, 80, 160, 320)
    @printf("    N=%4d  S=%9.1f   (2/pi)^2 N^2 = %9.1f\n", n, lam_k(n, n - 1) / lam_k(n, 1), (2 / pi)^2 * n^2)
end
