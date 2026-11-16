# rk45_scipy.jl — hand-written Dormand–Prince 5(4) (FSAL) that mirrors SciPy's solve_ivp(method="RK45")
# step by step: same initial-step heuristic, RMS error norm, safety 0.9, factor limits [0.2, 10],
# "no growth right after a rejection" rule, and nfev bookkeeping (2 + 6 per attempted step).
# Used by Lab 2.3 (RK45 nfev count), Lab 2.4 (Kepler) and Lab 2.6 (pendulum DAE).
using LinearAlgebra

const DP_C = (0.0, 1/5, 3/10, 4/5, 8/9, 1.0)
const DP_A = ((),
              (1/5,),
              (3/40, 9/40),
              (44/45, -56/15, 32/9),
              (19372/6561, -25360/2187, 64448/6561, -212/729),
              (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656))
const DP_B = (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84)
const DP_E = (-71/57600, 0.0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40)   # b5 - b4 (7 stages)

rmsnorm(v) = sqrt(sum(abs2, v) / length(v))      # SciPy's norm(x)/sqrt(n)

"""
    rk45(f, t0, y0, T; rtol, atol) -> (ts, ys, nfev, nrej)

`f(t, y)` returns dy/dt as a Vector{Float64}. `ts` are accepted step times, `ys[k]` the states.
"""
function rk45(f, t0, y0, T; rtol=1e-3, atol=1e-6)
    t = float(t0); y = collect(float.(y0)); fy = f(t, y); nfev = 1
    # --- initial step (SciPy select_initial_step, error order 4) ---
    sc = atol .+ abs.(y) .* rtol
    d0 = rmsnorm(y ./ sc); d1 = rmsnorm(fy ./ sc)
    h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1
    h0 = min(h0, T - t0)
    f1 = f(t + h0, y .+ h0 .* fy); nfev += 1
    d2 = rmsnorm((f1 .- fy) ./ sc) / h0
    h1 = (d1 <= 1e-15 && d2 <= 1e-15) ? max(1e-6, h0 * 1e-3) : (0.01 / max(d1, d2))^(1/5)
    habs = min(100 * h0, h1, T - t0)

    ts = [t]; ys = [copy(y)]; nrej = 0
    K = Vector{Vector{Float64}}(undef, 7)
    while t < T
        habs = max(habs, 10 * abs(nextfloat(t) - t))           # SciPy min_step guard
        rejected = false
        local tnew, ynew, fnew
        while true
            tnew = min(t + habs, T); h = tnew - t; habs = abs(h)
            K[1] = fy
            for s in 2:6
                ys_ = copy(y)
                for j in 1:s-1
                    ys_ .+= h * DP_A[s][j] .* K[j]
                end
                K[s] = f(t + DP_C[s] * h, ys_)
            end
            ynew = copy(y)
            for j in 1:6
                ynew .+= h * DP_B[j] .* K[j]
            end
            fnew = f(t + h, ynew); K[7] = fnew; nfev += 6       # FSAL: K[7] is next step's K[1]
            sc = atol .+ max.(abs.(y), abs.(ynew)) .* rtol
            errv = zeros(length(y))
            for j in 1:7
                errv .+= h * DP_E[j] .* K[j]
            end
            err = rmsnorm(errv ./ sc)
            if err < 1
                fac = err == 0 ? 10.0 : min(10.0, 0.9 * err^(-0.2))
                rejected && (fac = min(1.0, fac))
                habs *= fac
                break
            end
            habs *= max(0.2, 0.9 * err^(-0.2)); rejected = true; nrej += 1
        end
        t = tnew; y = ynew; fy = fnew
        push!(ts, t); push!(ys, copy(y))
    end
    return ts, ys, nfev, nrej
end
