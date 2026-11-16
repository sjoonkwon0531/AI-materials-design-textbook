# Lab 2.12 — implicit BTCS is unconditionally stable + recursive matrix solution U_n = D^n U_0
# (Julia port of the chapter code / lab2_11.py second half; h = k = 0.25, lambda = 4)
using LinearAlgebra, Printf

lam = 4.0
A = [9.0 -4.0 0.0; -4.0 9.0 -4.0; 0.0 -4.0 9.0]       # (1+2lam) diagonal, -lam off-diagonal
b = sin.(pi .* [0.25, 0.5, 0.75])                    # RHS = u^n (IC)
u1 = A \ b                                           # one implicit step (§1.4)
println("RHS = sin(pi x)       = ", round.(b, digits=4))
println("solution u(., k=0.25) = ", round.(u1, digits=5), "   <- all positive, physical")

D = inv(A); U = copy(b)
for _ in 1:5
    global U = A \ U                                 # five implicit steps by repeated solves
end
@printf("5 implicit steps: loop == D^5 * U0 ?  %s   (max diff %.1e)\n", isapprox(U, D^5 * b), maximum(abs.(U - D^5 * b)))
@printf("spectral radius rho(D_implicit) = %.3f\n", maximum(abs.(eigvals(D))))
T = [-2.0 1.0 0.0; 1.0 -2.0 1.0; 0.0 1.0 -2.0]          # discrete Laplacian (h^2 units)
for l in (0.4, 4.0)
    rho = maximum(abs.(eigvals(Symmetric(I + l * T))))
    @printf("explicit D = I + %.1f*T : rho = %.3f -> %s\n", l, rho, rho <= 1 ? "STABLE" : "UNSTABLE")
end
