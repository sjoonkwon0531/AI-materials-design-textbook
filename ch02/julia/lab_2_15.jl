# Lab 2.15 — minimal PINN for -u'' = pi^2 sin(pi x), u(0) = u(1) = 0 (same problem as Lab 2.8)
# u_theta(x) = sum_j v_j tanh(w_j x + b_j) + c, H = 20 (61 params), 80 collocation points, BC weight sqrt(10).
# u'' is analytic (tanh'' = -2 tanh (1 - tanh^2)); the residual Jacobian comes from ForwardDiff.jl and the
# least-squares training is a small hand-written Levenberg–Marquardt loop (SciPy's MINPACK 'lm' is not
# reproduced exactly). Julia's RNG differs from numpy's, so the final numbers agree only in range.
using ForwardDiff, LinearAlgebra, Random, Printf

const H = 20
function fwd(p, x)                                   # returns (u_theta(x), u_theta''(x))
    w, b, v, c = p[1:H], p[H+1:2H], p[2H+1:3H], p[3H+1]
    a = tanh.(x .* w' .+ b')                         # length(x) × H  (outer product as broadcast)
    return a * v .+ c, (-2 .* a .* (1 .- a .^ 2)) * (v .* w .^ 2)
end
const xc = collect(range(0, 1, length=80)); const f = pi^2 .* sin.(pi .* xc)
function residual(p)                                 # physics residual + weighted boundary residual
    u, uxx = fwd(p, xc)
    u0, _ = fwd(p, [0.0]); u1, _ = fwd(p, [1.0])
    return vcat(-uxx .- f, sqrt(10) .* [u0[1], u1[1]])
end

function levenberg_marquardt(r, p; maxiter=3000, xtol=1e-10, ftol=1e-15)
    mu = 1e-3; rp = r(p); cost = 0.5 * sum(abs2, rp); it = 0
    for outer it in 1:maxiter
        J = ForwardDiff.jacobian(r, p)               # forward-mode AD (§1.8)
        g = J' * rp; A = J' * J; d = diag(A)
        D = Diagonal(max.(d, 1e-10 * maximum(d)))    # Marquardt scaling
        local dp, pn, rn, cn
        while true
            dp = -((A + mu * D) \ g)
            pn = p + dp; rn = r(pn); cn = 0.5 * sum(abs2, rn)
            cn < cost && break                       # accept
            mu *= 4.0                                # reject: more gradient-descent-like
            mu > 1e20 && return p, cost, it
        end
        rel = (cost - cn) / cost
        p, rp, cost = pn, rn, cn; mu = max(mu / 3.0, 1e-12)   # accept: more Gauss–Newton-like
        (norm(dp) < xtol * (norm(p) + xtol) || rel < ftol) && break
    end
    return p, cost, it
end

p0 = 0.5 .* randn(Xoshiro(0), 3H + 1)                # initial params (statistically like the Python seed)
p, cost, iters = levenberg_marquardt(residual, p0)   # minimize the physics residual
xf = range(0, 1, length=1001); uf, _ = fwd(p, collect(xf))
err = maximum(abs.(uf .- sin.(pi .* xf)))
@printf("hidden units H=%d, collocation pts=%d, params=%d, LM iterations=%d\n", H, length(xc), length(p0), iters)
@printf("final residual cost (0.5 sum r^2) = %.3g\n", cost)
@printf("max |u_PINN - u_exact|            = %.3g\n", err)
@printf("u_PINN(0)=%.2g, u_PINN(1)=%.2g   (BC satisfied)\n", fwd(p, [0.0])[1][1], fwd(p, [1.0])[1][1])
println("NOTE: trained WITHOUT any labeled solution values - only the PDE + BC.")
