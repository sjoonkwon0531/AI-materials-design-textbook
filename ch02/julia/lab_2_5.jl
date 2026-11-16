# Lab 2.5 — shooting method for the steady fin/rod BVP  T'' = κ (T - Ta),  T(0)=40, T(10)=200
# (Julia port of lab2_5.py: RK4 IVP integrator ∘ bisection on the unknown slope T'(0))
using LinearAlgebra, Printf

const kap = 0.01; const Ta = 20.0; const L = 10.0; const T0 = 40.0; const TL = 200.0
rhs(Y) = [Y[2], kap * (Y[1] - Ta)]                        # [T', T'']
function shoot(slope; dx=0.01, full=false)               # RK4 integrate the IVP
    Y = [T0, slope]; prof = [Y[1]]
    for _ in 1:round(Int, L / dx)
        k1 = rhs(Y); k2 = rhs(Y + dx/2 * k1); k3 = rhs(Y + dx/2 * k2); k4 = rhs(Y + dx * k3)
        Y = Y + dx/6 * (k1 + 2k2 + 2k3 + k4); full && push!(prof, Y[1])
    end
    return full ? prof : Y[1]                             # T(L) for this slope
end

# analytic solution T = C1 e^{mx} + C2 e^{-mx} + Ta,  m = sqrt(κ)
m = sqrt(kap); c = [1.0 1.0; exp(m * L) exp(-m * L)] \ [T0 - Ta, TL - Ta]
Tex(x) = c[1] * exp(m * x) + c[2] * exp(-m * x) + Ta
@printf("analytic C1=%.4f C2=%.4f  T'(0)=%.6f\n", c[1], c[2], m * (c[1] - c[2]))

let I1 = 0.0, I2 = 500.0, s = 0.0, firsthit = 0       # bisection on the slope (§1.6)
    println("iter   slope guess    T(L) hit      error")
    for it in 1:50
        s = 0.5 * (I1 + I2); TLs = shoot(s); err = TLs - TL
        (firsthit == 0 && abs(err) < 1e-8) && (firsthit = it)
        it in (1, 5, 20, 39) && @printf("%4d  %12.6f  %11.6f  %+.2e\n", it, s, TLs, err)
        TLs < TL ? (I1 = s) : (I2 = s)
    end
    @printf("first iteration with |error| < 1e-8: %d\n", firsthit)
    @printf("converged slope T'(0) = %.6f\n", s)
    x = range(0, L, length=round(Int, L / 0.01) + 1)
    @printf("max |T_shoot - T_exact| over the rod = %.1e\n", maximum(abs.(shoot(s; full=true) .- Tex.(x))))
end

# linear BVP shortcut: g(s) = T(L;s) - 200 is linear in s -> one secant step hits the target
g0 = shoot(0.0) - TL; g1 = shoot(500.0) - TL
ssec = 0.0 - g0 * (500.0 - 0.0) / (g1 - g0)
@printf("secant (1 step): slope = %.6f   residual = %.1e\n", ssec, shoot(ssec) - TL)
