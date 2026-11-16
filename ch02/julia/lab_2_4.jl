# Lab 2.4 — e = 0.8 Kepler orbit with adaptive RK45 (hand-written Dormand–Prince, rk45_scipy.jl)
# GM = 1, a = 1, period 2π, start at perihelion r = 0.2 with speed 3; rtol = 1e-8, atol = 1e-11; 4 periods
using LinearAlgebra, Printf, Statistics
include("rk45_scipy.jl")

e = 0.8; P = 2pi
y0 = [1 - e, 0.0, 0.0, sqrt((1 + e) / (1 - e))]          # (x, y, vx, vy) = (0.2, 0, 0, 3)
function kepler(t, Y)
    r3 = (Y[1]^2 + Y[2]^2)^1.5
    return [Y[3], Y[4], -Y[1] / r3, -Y[2] / r3]
end
energy(Y) = 0.5 * (Y[3]^2 + Y[4]^2) - 1 / hypot(Y[1], Y[2])

ts, ys, nfev, nrej = rk45(kepler, 0.0, y0, 4P; rtol=1e-8, atol=1e-11)
dts = diff(ts)
@printf("accepted steps = %d   nfev = %d   rejected = %d\n", length(dts), nfev, nrej)
for k in 0:3                                              # step-size range in each orbit
    m = [k * P <= ts[i] < (k + 1) * P for i in eachindex(dts)]
    d = dts[m]
    @printf("  period %d: dt min %.4f  max %.4f  ratio %.0f\n", k + 1, minimum(d), maximum(d), maximum(d) / minimum(d))
end
r = [hypot(Y[1], Y[2]) for Y in ys]
@printf("median dt near perihelion (r<0.25) = %.4f ; near aphelion (r>1.75) = %.4f\n",
        median(dts[r[1:end-1] .< 0.25]), median(dts[r[1:end-1] .> 1.75]))
@printf("position error after 4 periods |Y(T)-Y0| = %.2e\n", norm(ys[end][1:2] - y0[1:2]))
@printf("energy E0=%.8f  E_end=%.8f  rel drift=%.2e   (adaptive != symplectic)\n",
        energy(ys[1]), energy(ys[end]), energy(ys[end]) / energy(ys[1]) - 1)
