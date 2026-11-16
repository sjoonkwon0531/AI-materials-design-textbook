from scipy.integrate import quad
import numpy as np

# 페르미 분포의 적분 (페르미 표면 근처에서 가파름)
def fermi(E, E_F=1.0, kT=0.025):
    return 1.0 / (np.exp((E - E_F)/kT) + 1.0)

result, error = quad(fermi, 0, 2.0)
print(f"Integral = {result:.6f}, est. error = {error:.2e}")
