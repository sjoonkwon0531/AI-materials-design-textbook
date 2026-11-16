# Lab 2.1 — explicit Euler / implicit Euler / RK4 vs exact exp(A t) X0 (Julia port of lab2_1.py)
using LinearAlgebra, Printf

A  = [-1.0  0.5  0.0;
       1.0 -1.5  0.3;
       0.0  1.0 -0.3]                       # mass-conserving: column sums = 0
x0 = [1.0, 0.0, 0.0]; T = 5.0
println("col sums = ", vec(sum(A, dims=1)), "   eig = ", sort(real.(eigvals(A))))
exact(t) = exp(A * t) * x0                   # matrix exponential (LinearAlgebra)

fe(x, h, A)  = x + h * (A * x)               # explicit Euler
be(x, h, A)  = (I - h * A) \ x               # implicit Euler: one linear solve (§1.4)
function rk4(x, h, A)                        # classical RK4 (§2.2.3)
    k1 = A * x; k2 = A * (x + h/2 * k1); k3 = A * (x + h/2 * k2); k4 = A * (x + h * k3)
    return x + h/6 * (k1 + 2k2 + 2k3 + k4)
end
function march(step, h)
    x = copy(x0)
    for _ in 1:round(Int, T / h)
        x = step(x, h, A)
    end
    return x
end

xe = exact(T)
for (name, s) in [("FE", fe), ("BE", be), ("RK4", rk4)]
    prev = NaN
    for h in (0.1, 0.05, 0.025, 0.0125)
        e = norm(march(s, h) - xe)
        @printf("%-4s h=%-7g err(t=5)=%.3e  %s\n", name, h, e, isnan(prev) ? "" : @sprintf("ratio %.2f", prev / e))
        prev = e
    end
end

# stiffen: X1 -> X2 rate constant 1 -> 1000
As = [-1000.0 0.5 0.0; 1000.0 -1.5 0.3; 0.0 1.0 -0.3]
lam = eigvals(As); lmax = maximum(abs.(real.(lam)))
println("stiff eig = ", sort(real.(lam)), "   2/|lmax| = ", 2 / lmax)
for h in (0.0019, 0.0021)
    x = copy(x0)
    for _ in 1:round(Int, T / h)
        x = x + h * (As * x)
    end
    @printf("FE stiff h=%.4f  max|x(5)| = %.3e\n", h, maximum(abs.(x)))
end
