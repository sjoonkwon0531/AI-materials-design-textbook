# Lab 4.2 (Julia) — 가장 단순한 1D Kohn–Sham SCF 루프 (선형 혼합 = 댐핑된 고정점 반복)
# 파이썬 판(lab_4_2.py)과 같은 모형: 조화트랩, 전자 4개(오비탈 2개 이중점유),
# 모형 Hartree(soft-Coulomb, a=1) + 모형 국소교환 E_x = -cx ∫ n^2.
using LinearAlgebra, Printf

L, N = 8.0, 400
x = collect(range(-L, L; length = N)); dx = x[2] - x[1]
Vext = 0.5 .* x .^ 2
T = -0.5 .* diagm(0 => fill(-2.0, N), 1 => ones(N - 1), -1 => ones(N - 1)) ./ dx^2   # Lab 4.1의 FDM 운동에너지
const Nelec, nocc = 4, 2
const a_soft, cx = 1.0, 0.30
Kmat = @. 1.0 / sqrt((x - x')^2 + a_soft^2)        # x - x' : 열벡터-행벡터 브로드캐스팅 = N x N

function density(psi)
    n = zeros(N)
    for i in 1:nocc                                  # 1-based: 오비탈 1, 2
        n .+= 2.0 .* (psi[:, i] ./ sqrt(dx)) .^ 2
    end
    return n
end
potentials(n, lamH) = (lamH .* (Kmat * n) .* dx, -2.0 * cx .* n)     # (v_H, v_x)

function total_energy(eps, n, lamH)
    vH, vx = potentials(n, lamH)
    EH = 0.5 * sum(n .* vH) * dx
    Ex = -cx * sum(n .^ 2) * dx
    band = 2.0 * sum(eps[1:nocc])
    return band - EH - sum(vx .* n) * dx + Ex
end

function scf(lamH, alpha; n_iter = 200, tol = 1e-9, verbose = false)
    n = exp.(-x .^ 2); n .*= Nelec / (sum(n) * dx)   # 초기 밀도
    Ehist = Float64[]; dnhist = Float64[]
    eps = psi = nothing
    for it in 0:n_iter-1                             # it는 파이썬과 같은 0부터 (출력용)
        vH, vx = potentials(n, lamH)
        F = eigen(Symmetric(T + Diagonal(Vext .+ vH .+ vx)))   # ← Lab 4.1 솔버 재사용
        eps, psi = F.values, F.vectors
        n_new = density(psi); n_new .*= Nelec / (sum(n_new) * dx)
        dn = sqrt(sum((n_new .- n) .^ 2) * dx)
        push!(Ehist, total_energy(eps, n_new, lamH)); push!(dnhist, dn)
        if verbose && it < 12
            @printf("  it %2d  E=%10.6f  |dn|=%.3e  eps0=%.5f\n", it, Ehist[end], dn, eps[1])
        end
        dn < tol && break
        n = (1 - alpha) .* n .+ alpha .* n_new       # 선형 혼합(댐핑)
    end
    return Ehist, dnhist, n, eps, psi
end

println("--- (1) Weak coupling lamH=0.5, mixing alpha=0.3 ---")
Eh, dnh, n_conv, epsKS, psiKS = scf(0.5, 0.3; verbose = true)
@printf("\n  converged in %d iters,  E = %.6f\n", length(Eh), Eh[end])
@printf("  KS eigenvalues (lowest 4): [%.4f %.4f %.4f %.4f]\n", epsKS[1:4]...)
@printf("  electron count = %.6f (target %d)\n", sum(n_conv) * dx, Nelec)

println("\n--- (2) Strong coupling lamH=4.0 (200 iterations, no early stop) ---")
println(" alpha   first |dn|<1e-9   |dn| after 200   verdict")
for al in (1.0, 0.5, 0.4, 0.3, 0.15)
    _, dn, _ = scf(4.0, al; n_iter = 200, tol = 0.0)
    hit = findfirst(<(1e-9), dn)                     # 1-based 인덱스 = 반복 횟수 (파이썬의 hit[0]+1)
    firststr = hit === nothing ? " never" : @sprintf("%6d", hit)
    verdict = dn[end] < 1e-9 ? "converged" :
              (hit === nothing ? "sloshing (no convergence)" : "transient dip, then diverges")
    @printf(" %4.2f   %15s   %12.2e   %s\n", al, firststr, dn[end], verdict)
end
