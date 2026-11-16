# Lab 2.2 — undamped mass-spring: energy under explicit Euler, implicit Euler, RK4, velocity-Verlet
# (Julia port of lab2_2.py; m = k = 1, x0 = 1, v0 = 0, dt = 0.1)
using LinearAlgebra, Printf

const w2 = 1.0; const x0 = 1.0; const v0 = 0.0; const dt = 0.1
energy(x, v) = 0.5 * v^2 + 0.5 * w2 * x^2            # exact E0 = 0.5

function simulate(T)
    N = round(Int, T / dt); out = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}()
    # explicit Euler
    X = zeros(N + 1); V = zeros(N + 1); X[1], V[1] = x0, v0
    for n in 1:N
        X[n+1] = X[n] + dt * V[n]; V[n+1] = V[n] - dt * w2 * X[n]
    end
    out["explicit Euler"] = (X, V)
    # implicit Euler: [1 -dt; dt*w2 1] [x;v]_{n+1} = [x;v]_n
    M = inv([1.0 -dt; dt*w2 1.0])
    X = zeros(N + 1); V = zeros(N + 1); X[1], V[1] = x0, v0
    for n in 1:N
        X[n+1], V[n+1] = M * [X[n], V[n]]
    end
    out["implicit Euler"] = (X, V)
    # RK4
    f(y) = [y[2], -w2 * y[1]]
    X = zeros(N + 1); V = zeros(N + 1); X[1], V[1] = x0, v0; y = [x0, v0]
    for n in 1:N
        k1 = f(y); k2 = f(y + dt/2 * k1); k3 = f(y + dt/2 * k2); k4 = f(y + dt * k3)
        y = y + dt/6 * (k1 + 2k2 + 2k3 + k4); X[n+1], V[n+1] = y
    end
    out["RK4"] = (X, V)
    # velocity-Verlet (Box 2.6)
    X = zeros(N + 1); V = zeros(N + 1); X[1], V[1] = x0, v0; x, v = x0, v0; a = -w2 * x
    for n in 1:N
        xn = x + v * dt + 0.5 * a * dt^2; an = -w2 * xn
        v = v + 0.5 * (a + an) * dt; x = xn; a = an; X[n+1], V[n+1] = x, v
    end
    out["velocity-Verlet"] = (X, V)
    return out
end

o = simulate(30.0)
@printf("T=30 (%.2f periods)\n", 30 / (2pi))
for k in ("explicit Euler", "implicit Euler", "RK4", "velocity-Verlet")
    X, V = o[k]; e = energy.(X, V)
    @printf("  %-16s E(30)=%.6f  E/E0=%.4f  max|x|=%.3f  E in [%.4f, %.4f]\n",
            k, e[end], e[end] / 0.5, maximum(abs.(X)), minimum(e), maximum(e))
end
ol = simulate(2000.0)
@printf("T=2000 (%.2f periods, %d steps)\n", 2000 / (2pi), round(Int, 2000 / dt))
for k in ("RK4", "velocity-Verlet")
    X, V = ol[k]; e = energy.(X, V)
    @printf("  %-16s E_end=%.6f  drift=%.4f%%  E in [%.6f, %.6f]  band=%.3e\n",
            k, e[end], 100 * (e[end] / 0.5 - 1), minimum(e), maximum(e), maximum(e) - minimum(e))
end
@printf("explicit-Euler area factor per step = %.2f, after 300 steps = %.2f\n", 1 + dt^2, (1 + dt^2)^300)
