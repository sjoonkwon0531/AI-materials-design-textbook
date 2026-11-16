# Lab 2.6 — pendulum DAE (index-1 reduction): naive vs Baumgarte stabilization (Julia port of lab2_6.py)
# m = L = 1, g = 9.81, horizontal release (x,y,vx,vy) = (1,0,0,0); RK45 rtol = 1e-7, atol = 1e-6; alpha = beta = 5
using Printf
include("rk45_scipy.jl")

const g = 9.81
function make(alpha, beta)
    return function (t, u)
        x, y, vx, vy = u
        r2 = x^2 + y^2; v2 = vx^2 + vy^2
        phi = r2 - 1; dphi = 2 * (x * vx + y * vy)                       # constraint and its rate
        lam = (v2 - g * y + alpha * dphi + 0.5 * beta^2 * phi) / r2      # multiplier (tension / mass)
        return [vx, vy, -lam * x, -lam * y - g]
    end
end
u0 = [1.0, 0.0, 0.0, 0.0]; T = 20.0
viol(u) = abs(u[1]^2 + u[2]^2 - 1)

res = Dict{String,Any}()
for (name, ab) in (("naive", 0.0), ("Baumgarte", 5.0))
    ts, us, nfev, _ = rk45(make(ab, ab), 0.0, u0, T; rtol=1e-7, atol=1e-6)
    v = viol.(us)
    res[name] = (ts, v)
    @printf("%-10s |x^2+y^2-1| at t=20 = %.2e   max over accepted steps = %.2e   nfev = %d\n",
            name, v[end], maximum(v), nfev)
end
@printf("drift reduction at t=20: %.1e x\n", res["naive"][2][end] / res["Baumgarte"][2][end])
# 2-second-window maxima (SciPy version samples a dense output; here we use the accepted steps)
for name in ("naive", "Baumgarte")
    ts, v = res[name]
    w = [maximum(v[(ts .>= t0) .& (ts .< t0 + 1)]) for t0 in 0:2:18]
    println(rpad(name, 10), " 1-s window max every 2 s: ", join([@sprintf("%.1e", q) for q in w], " "))
end
