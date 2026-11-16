# Lab 2.13 — CFL check: upwind advection of a square wave (Julia port of lab2_13.py)
# periodic x in [0,1), N = 200 cells (dx = 0.005), a = 1, u = 1 on [0.1, 0.3), 300 steps
using Printf

N = 200; dx = 1.0 / N; x = collect(0:N-1) .* dx          # elementwise i*dx, as np.arange(N)*dx
u0 = Float64.((x .>= 0.1) .& (x .< 0.3))
function upwind(u, nu, steps)
    u = copy(u)
    for _ in 1:steps
        u = u .- nu .* (u .- circshift(u, 1))       # circshift(u,1) == np.roll(u,1)
    end
    return u
end
res = Dict{Float64,Vector{Float64}}()
for nu in (0.8, 1.0, 1.3)
    res[nu] = upwind(u0, nu, 300)
    @printf("CFL nu=%.1f : max|u| = %.6g\n", nu, maximum(abs.(res[nu])))
end
println("nu=1 is an exact shift by 300 cells: ", res[1.0] == circshift(u0, 300))
