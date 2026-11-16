# §1.9.4 (Julia): 2D Ising — Metropolis Monte Carlo
# Python(NumPy + Numba) 구현의 Julia 대응. Julia는 함수가 JIT 컴파일되므로 Numba가 필요 없다.
using Random, Printf, Statistics

"""한 sweep = L*L번의 spin flip 시도 (in-place)."""
function ising_mc_step!(spins::Matrix{Int}, beta::Float64; J = 1.0, h = 0.0, rng = Random.default_rng())
    L = size(spins, 1)
    for _ in 1:L*L
        i = rand(rng, 1:L)
        j = rand(rng, 1:L)
        # 주기 경계조건: mod1은 1..L 범위로 감는다 (1-based 인덱스)
        nb_sum = spins[mod1(i + 1, L), j] + spins[mod1(i - 1, L), j] +
                 spins[i, mod1(j + 1, L)] + spins[i, mod1(j - 1, L)]
        dE = 2J * spins[i, j] * nb_sum + 2h * spins[i, j]
        if dE <= 0 || rand(rng) < exp(-beta * dE)
            spins[i, j] = -spins[i, j]
        end
    end
    return spins
end

magnetization(spins) = abs(mean(spins))

# 시뮬레이션: T_c 근처에서 자화 측정
rng = MersenneTwister(0)      # Python 원본은 seed를 고정하지 않는다; 결과는 통계적으로만 일치
L = 32
spins = rand(rng, [-1, 1], L, L)
n_equilibrate = 1000
n_measure = 5000

T_list = range(1.5, 3.5; length = 21)
M_list = Float64[]

for T in T_list
    beta = 1.0 / T
    for _ in 1:n_equilibrate              # Equilibration
        ising_mc_step!(spins, beta; rng = rng)
    end
    M_samples = Float64[]
    for _ in 1:n_measure                  # Measurement
        ising_mc_step!(spins, beta; rng = rng)
        push!(M_samples, magnetization(spins))
    end
    push!(M_list, mean(M_samples))
end

# Onsager의 임계온도와 비교
T_c_exact = 2.0 / log(1 + sqrt(2))
@printf("Onsager T_c = %.4f\n", T_c_exact)
for (T, M) in zip(T_list, M_list)        # Python의 plot(T_list, M_list) 대신 표로 출력
    @printf("T = %.1f   <|m|> = %.4f\n", T, M)
end
