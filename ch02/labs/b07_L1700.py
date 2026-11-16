import numpy as np
lam = 4.0
A = np.array([[9.,-4.,0.],[-4.,9.,-4.],[0.,-4.,9.]])   # (1+2lam) diag, -lam off
b = np.sin(np.pi*np.array([0.25,0.5,0.75]))            # RHS = u^n (IC)
u1 = np.linalg.solve(A, b)                             # one implicit step (§1.4)
