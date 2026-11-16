# Lab 2.8 — P1 finite elements for -u'' = pi^2 sin(pi x), u(0) = u(1) = 0  (exact u = sin(pi x))
# (Julia port of the chapter code: stiffness matrix (1/h)[-1,2,-1], load vector by 2-point Gauss)
using LinearAlgebra, SparseArrays, Printf

exact(x) = sin(pi * x)
ffun(x)  = pi^2 * sin(pi * x)

function fem_p1(N)                          # N elements, N-1 interior unknowns
    h = 1.0 / N; x = collect(range(0, 1, length=N + 1)); nint = N - 1
    K = spdiagm(-1 => fill(-1 / h, nint - 1), 0 => fill(2 / h, nint), 1 => fill(-1 / h, nint - 1))
    b = zeros(N + 1); gp = (-1 / sqrt(3), 1 / sqrt(3))       # 2-point Gauss (§1.7)
    for e in 1:N                            # element e spans nodes e, e+1 (1-based)
        x0 = x[e]
        for xi in gp
            xg = x0 + 0.5 * h * (1 + xi)
            N0, N1, jac = 0.5 * (1 - xi), 0.5 * (1 + xi), 0.5 * h
            b[e]   += ffun(xg) * N0 * jac
            b[e+1] += ffun(xg) * N1 * jac
        end
    end
    u = zeros(N + 1); u[2:N] = K \ b[2:N]   # sparse direct solve
    return x, u, K
end

# L2 error of the P1 interpolant on a sub-grid with nf points per element, trapezoid rule
# (equivalent to np.interp + np.trapezoid in the Python version)
function l2_err(x, u; nf=20)
    N = length(x) - 1; xf = range(0, 1, length=N * nf + 1)
    uf = [begin
              e = min(floor(Int, xs * N) + 1, N)                 # element containing xs (uniform mesh)
              t = (xs - x[e]) / (x[e+1] - x[e]); (1 - t) * u[e] + t * u[e+1]
          end for xs in xf]
    d2 = (uf .- exact.(xf)) .^ 2; dx = step(xf)
    return sqrt(dx * (sum(d2) - 0.5 * (d2[1] + d2[end])))
end

@printf("%5s %8s %12s %12s %9s %11s\n", "N", "h", "max_err", "L2_err", "L2_order", "nodal_order")
let prev = NaN, pm = NaN
    for N in (4, 8, 16, 32, 64, 128, 256)
        x, u, _ = fem_p1(N); me = maximum(abs.(u .- exact.(x))); e = l2_err(x, u)
        o  = isnan(prev) ? "--" : @sprintf("%.3f", log2(prev / e))
        om = isnan(pm)   ? "--" : @sprintf("%.3f", log2(pm / me))
        @printf("%5d %8.5f %12.3e %12.3e %9s %11s\n", N, 1 / N, me, e, o, om)
        prev = e; pm = me
    end
end
_, _, K8 = fem_p1(8)
@printf("stiffness interior row (N=8): off = %.4f, diag = %.4f  (1/h = %.4f)  == h * FDM Laplacian\n",
        K8[4, 3], K8[4, 4], 8.0)
