# Lab 2.9 — first-order upwind finite volumes for u_t + u_x = 0 (a = 1) on periodic [0,1]
# (Julia port of the chapter code: square wave, N = 200, CFL = 0.8, one period t = 1, mass tracking)
using Printf

function run_fvm(; N=200, a=1.0, CFL=0.8, tfinal=1.0)
    dx = 1.0 / N; xc = (collect(0:N-1) .+ 0.5) .* dx
    u = Float64.((xc .> 0.3) .& (xc .< 0.6))          # square-wave IC
    u0 = copy(u); dt = CFL * dx / a; nu = a * dt / dx
    mass = [sum(u) * dx]; t = 0.0
    while t < tfinal - 1e-12
        if t + dt > tfinal
            dt = tfinal - t; nu = a * dt / dx
        end
        u = u .- nu .* (u .- circshift(u, 1))          # upwind; circshift(u,1) == np.roll(u,1)
        t += dt; push!(mass, sum(u) * dx)
    end
    return xc, u0, u, mass
end

xc, u0, u, mass = run_fvm()
@printf("initial mass = %.10f\n", mass[1])
@printf("final   mass = %.10f\n", mass[end])
@printf("max relative drift over run = %.3e   (%d records)\n", maximum(abs.(mass .- mass[1]) ./ mass[1]), length(mass))
@printf("max |u(1) - u0| = %.3f   (numerical diffusion smears the square wave)\n", maximum(abs.(u .- u0)))
_, _, ub, _ = run_fvm(CFL=1.05)
@printf("CFL = 1.05: max|u(1)| = %.3e   (nu > 1 -> unstable)\n", maximum(abs.(ub)))
