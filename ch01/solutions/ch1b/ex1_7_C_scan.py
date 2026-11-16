"""Ex 1.7.C supplement: error of tensor Gauss + indicator vs n (non-monotone)."""
import numpy as np
from ex1_7_C_lib import exact, tensor_gauss
for d, ns in [(2, [5,10,20,40,80,160,320]), (5, list(range(4, 41, 4))), (10, list(range(2, 21, 2)))]:
    I = exact(d)
    print(f"d={d}: " + ", ".join(f"n={n}:{abs(tensor_gauss(d,n)-I)/I:.1e}" for n in ns))
