# Lab 2.11 — three-node FTCS stability experiment, lambda = 0.4 vs lambda = 4 (Julia port of lab2_11.py)
using Printf

function ftcs_step(u, lam)
    un = copy(u)
    un[2:end-1] = u[2:end-1] + lam * (u[3:end] - 2u[2:end-1] + u[1:end-2])   # 1-based slices
    return un
end
fmt(v) = "[" * join([@sprintf("%.4f", q) for q in v], " ") * "]"

h = 0.25; x = collect(0:h:1)                     # 5 nodes, 3 interior
u0 = sin.(pi .* x); u0[1] = u0[end] = 0.0          # IC = sin(pi x)
println("IC sin(pi x)              = ", fmt(u0))
println("STABLE   lam=0.4 (k=0.025): ", fmt(ftcs_step(u0, 0.4)))
println("UNSTABLE lam=4   (k=0.25) : ", fmt(ftcs_step(u0, 4.0)))
@printf("stability bound: lam <= 1/2  ->  k <= h^2/2 = %.5f\n", h^2 / 2)
let u = copy(u0)
    for n in 1:6
        u = ftcs_step(u, 4.0); @printf("lam=4 step %d: max|u| = %.2f\n", n, maximum(abs.(u)))
    end
end
