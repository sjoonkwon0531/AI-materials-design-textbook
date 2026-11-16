# Lab 2.3 — stiff network: explicit Euler blow-up vs implicit BDF (Julia port of lab2_3.py)
# X1 -(1001.5)-> X2,  X2 <-> X3 (kf = 0.5, kr = 1.0),  X0 = (1,0,0),  0 <= t <= 8
# NOTE: SciPy's variable-order/variable-step BDF (NDF) is not re-implemented here. Instead we hand-write a
# FIXED-STEP BDF2 (first step by BDF1 = implicit Euler) and, for the cost comparison, the SciPy-faithful
# adaptive RK45 from rk45_scipy.jl.
using LinearAlgebra, Printf
include("rk45_scipy.jl")

A  = [-1001.5 0.0 0.0;
       1001.5 -0.5 1.0;
       0.0    0.5 -1.0]
x0 = [1.0, 0.0, 0.0]; T = 8.0
lam = sort(real.(eigvals(A)))
@printf("eig = %s   S = %.1f   2/|lmax| = %.6f\n", string(round.(lam, digits=4)), abs(lam[1]) / abs(lam[2]), 2 / abs(lam[1]))
exact(t) = exp(A * t) * x0

# (1) explicit Euler just below / above the stability limit
for dt in (0.0019, 0.0020, 0.0021, 0.0022)
    x = copy(x0)
    for _ in 1:round(Int, T / dt)
        x = x + dt * (A * x)
    end
    @printf("FE dt=%.4f  max|x(8)| = %.3e\n", dt, maximum(abs.(x)))
end

# (2) fixed-step BDF2:  (3x_{n+1} - 4x_n + x_{n-1})/(2h) = A x_{n+1}
#     => (I - 2h/3 A) x_{n+1} = (4x_n - x_{n-1})/3 ; factor once (lu), one triangular solve per step
function bdf2(h)
    n = round(Int, T / h)
    xm = copy(x0); x = (I - h * A) \ x0                 # start-up: one BDF1 (implicit Euler) step
    F = lu(I - (2h / 3) * A)
    for _ in 2:n
        xn = F \ ((4x - xm) / 3); xm = x; x = xn
    end
    return x, n
end
prev = NaN
for h in (0.1, 0.04, 0.02, 0.01)
    x, n = bdf2(h); e = maximum(abs.(x - exact(T)))
    @printf("BDF2 h=%-5g (h|lmax|=%6.1f) steps=%4d  err(t=8)=%.2e  %s\n", h, h * abs(lam[1]), n, e,
            isnan(prev) ? "" : @sprintf("ratio %.2f", prev / e))
    global prev = e
end

# (3) cost of a non-stiff (explicit, adaptive) solver on the same problem: RK45, rtol=1e-6, atol=1e-9
ts, ys, nfev, nrej = rk45((t, x) -> A * x, 0.0, x0, T; rtol=1e-6, atol=1e-9)
@printf("RK45 (rtol=1e-6, atol=1e-9): steps=%d  nfev=%d  rejected=%d  err(t=8)=%.1e\n",
        length(ts) - 1, nfev, nrej, maximum(abs.(ys[end] - exact(T))))
