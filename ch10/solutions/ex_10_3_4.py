"""Ex 10.3.4: prevalence and DAF."""
n, s = 215488, 32942
p = s / n
print(f"prevalence = {p:.5f}; DAF at precision 0.6 = {0.6/p:.3f}; max DAF (precision 1) = {1/p:.3f}")
