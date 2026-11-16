# Lab 4.4 (Julia) — Velocity Verlet로 Lennard-Jones MD 돌리기
#  Part A: 조화진동자에서 심플렉틱성(Verlet / symplectic Euler / forward Euler) — 결정론적, 숫자 일치
#  Part B: 2D LJ NVE (N=36, rho=0.7) 에너지 보존       — 난수 초기속도: 통계적으로만 일치
#  Part C: 속도 재조정 NVT (T0=1.0)                     — 난수 초기속도: 통계적으로만 일치
# 환산 LJ 단위(epsilon=sigma=m=1), PBC + 최소영상규약.
# 배열 배치: Julia는 column-major이므로 좌표를 2 x N (열 = 입자)로 둔다(파이썬은 N x 2).
using LinearAlgebra, Printf, Random, Statistics
Random.seed!(1)

# ---------------- Part A ----------------
function sho_run(method, dt, nsteps; x0 = 1.0, v0 = 0.0)
    x, v = x0, v0
    E = [0.5 * (v * v + x * x)]
    for _ in 1:nsteps
        if method == :verlet
            v += 0.5 * dt * (-x); x += dt * v; v += 0.5 * dt * (-x)
        elseif method == :euler                      # forward Euler: 옛 x, v로 동시에 갱신
            x, v = x + dt * v, v + dt * (-x)
        elseif method == :symplectic_euler
            v += dt * (-x); x += dt * v
        end
        push!(E, 0.5 * (v * v + x * x))
    end
    return E
end
dt_sho, n_sho = 0.15, 300
Ev = sho_run(:verlet, dt_sho, n_sho)
Ee = sho_run(:euler, dt_sho, n_sho)
Es = sho_run(:symplectic_euler, dt_sho, n_sho)
println("=== Part A: harmonic oscillator, energy after ", n_sho, " steps (exact=0.5) ===")
@printf("  Velocity Verlet : E_end=%.4f   drift=%+.2e\n", Ev[end], (Ev[end] - Ev[1]) / Ev[1])
@printf("  Symplectic Euler: E_end=%.4f   drift=%+.2e\n", Es[end], (Es[end] - Es[1]) / Es[1])
@printf("  Forward Euler   : E_end=%.4f   drift=%+.2e  (blows up)\n", Ee[end], (Ee[end] - Ee[1]) / Ee[1])

# ---------------- Part B/C: 2D LJ ----------------
const rc = 2.5; const rc2 = rc * rc; const ushift = 4.0 * (rc^-12 - rc^-6)

function init_lattice(nside, density)
    N = nside * nside; L = sqrt(N / density); a = L / nside
    xs = ((0:nside-1) .+ 0.5) .* a
    pos = zeros(2, N)
    k = 0
    for i in 1:nside, j in 1:nside                   # 파이썬 meshgrid + ravel과 같은 순서
        k += 1
        pos[1, k] = xs[j]; pos[2, k] = xs[i]
    end
    return pos, L, N
end

"LJ 힘과 퍼텐셜에너지 (절단 + 에너지 이동, 최소영상규약)"
function forces_energy(pos, L)
    N = size(pos, 2); F = zeros(2, N); PE = 0.0
    for i in 1:N-1, j in i+1:N
        dx = pos[1, i] - pos[1, j]; dy = pos[2, i] - pos[2, j]
        dx -= L * round(dx / L); dy -= L * round(dy / L)   # minimum image (round: ties-to-even, numpy와 동일)
        r2 = dx * dx + dy * dy
        if r2 < rc2
            inv2 = 1.0 / r2; inv6 = inv2^3
            fs = (48.0 * inv6 * inv6 - 24.0 * inv6) * inv2
            F[1, i] += fs * dx; F[2, i] += fs * dy
            F[1, j] -= fs * dx; F[2, j] -= fs * dy
            PE += 4.0 * (inv6 * inv6 - inv6) - ushift
        end
    end
    return F, PE
end

function kinetic_temp(vel)
    KE = 0.5 * sum(abs2, vel); N = size(vel, 2)
    return KE, 2.0 * KE / (2N - 2)                   # 2D, 총운동량 0 제약: 자유도 2N-2
end

function run_md(nsteps, dt; nside = 6, density = 0.7, T0 = 1.0, thermostat = nothing)
    pos, L, N = init_lattice(nside, density)
    vel = randn(2, N); vel .-= mean(vel; dims = 2)  # 총운동량 0
    _, T = kinetic_temp(vel); vel .*= sqrt(T0 / T)
    F, PE = forces_energy(pos, L)
    h = Dict(k => Float64[] for k in ("t", "KE", "PE", "E", "T"))
    for s in 0:nsteps-1
        vel .+= 0.5 * dt .* F                        # kick
        pos = mod.(pos .+ dt .* vel, L)              # drift  (mod: 파이썬 %와 같이 [0,L)로; rem 아님)
        F, PE = forces_energy(pos, L)
        vel .+= 0.5 * dt .* F                        # kick
        KE, T = kinetic_temp(vel)
        if thermostat == :rescale && s % 20 == 0 && T > 0
            vel .*= sqrt(T0 / T); KE, T = kinetic_temp(vel)
        end
        if s % 5 == 0
            push!(h["t"], s * dt); push!(h["KE"], KE); push!(h["PE"], PE)
            push!(h["E"], KE + PE); push!(h["T"], T)
        end
    end
    return h
end

dt, nsteps = 0.004, 6000
println("\n=== Part B: 2D LJ NVE (Velocity Verlet), N=36, rho=0.7 ===")
nve = run_md(nsteps, dt)
E = nve["E"]
pp = (maximum(E) - minimum(E)) / abs(mean(E))
@printf("  total E mean=%.3f, peak-to-peak/|E|=%.2e, E(end)/E(0)=%.4f\n", mean(E), pp, E[end] / E[1])
println("=== Part C: NVT velocity-rescaling thermostat, T0=1.0 ===")
nvt = run_md(nsteps, dt; thermostat = :rescale, T0 = 1.0)
half = length(nvt["T"]) ÷ 2                          # 정수 나눗셈
Tn = nvt["T"][half+1:end]                            # 파이썬 T[half:]
@printf("  NVT mean T (2nd half)=%.3f +/- %.3f  (target 1.000)\n", mean(Tn), std(Tn; corrected = false))  # np.std = 모표준편차
@printf("  NVE mean T (2nd half)=%.3f  (uncontrolled)\n", mean(nve["T"][half+1:end]))
