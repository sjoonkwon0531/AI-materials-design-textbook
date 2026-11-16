import numpy as np
from numba import njit

@njit
def ising_mc_step(spins, beta, J=1.0, h=0.0):
    """한 sweep = L*L번의 spin flip 시도."""
    L = spins.shape[0]
    for _ in range(L*L):
        i = np.random.randint(L)
        j = np.random.randint(L)
        # 주기 경계조건: 4개 이웃의 합
        nb_sum = (spins[(i+1)%L, j] + spins[(i-1)%L, j] 
                + spins[i, (j+1)%L] + spins[i, (j-1)%L])
        dE = 2 * J * spins[i, j] * nb_sum + 2 * h * spins[i, j]
        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            spins[i, j] *= -1
    return spins

@njit
def magnetization(spins):
    return np.abs(np.mean(spins))

# 시뮬레이션: T_c 근처에서 자화 측정
L = 32  # 격자 크기
spins = np.random.choice([-1, 1], size=(L, L))
n_equilibrate = 1000  # equilibration
n_measure = 5000     # measurement sweeps

T_list = np.linspace(1.5, 3.5, 21)
M_list = []

for T in T_list:
    beta = 1.0 / T
    # Equilibration
    for _ in range(n_equilibrate):
        spins = ising_mc_step(spins, beta)
    # Measurement
    M_samples = []
    for _ in range(n_measure):
        spins = ising_mc_step(spins, beta)
        M_samples.append(magnetization(spins))
    M_list.append(np.mean(M_samples))

# Onsager의 임계온도와 비교
T_c_exact = 2.0 / np.log(1 + np.sqrt(2))
print(f"Onsager T_c = {T_c_exact:.4f}")
# plot(T_list, M_list)로 그리면 T_c 근처에서 자화가 0으로 떨어지는 것이 보임