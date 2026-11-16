import numpy as np
np.set_printoptions(precision=4, suppress=True)
def ftcs_step(u, lam):
    un = u.copy()
    un[1:-1] = u[1:-1] + lam*(u[2:] - 2*u[1:-1] + u[:-2])
    return un
h = 0.25; x = np.arange(0, 1+1e-9, h)
u0 = np.sin(np.pi*x); u0[0] = u0[-1] = 0
print("IC", u0)
print("stable 0.4", ftcs_step(u0,0.4))
print("unstable 4", ftcs_step(u0,4.0))
print("bound k<=h^2/2", h*h/2)
u=u0.copy()
for n in range(1,7):
    u=ftcs_step(u,4.0); print(n, np.abs(u).max(), u)
print("lam=0.6 one step", ftcs_step(u0,0.6))
# Lab 2.12
lam=4.0
A=np.array([[9.,-4.,0.],[-4.,9.,-4.],[0.,-4.,9.]])
b=np.sin(np.pi*np.array([0.25,0.5,0.75]))
u1=np.linalg.solve(A,b); print("implicit", np.round(u1,5))
D=np.linalg.inv(A); U=b.copy()
for i in range(5): U=np.linalg.solve(A,U)
print("D^5 equal", np.allclose(U, np.linalg.matrix_power(D,5)@b), np.abs(U-np.linalg.matrix_power(D,5)@b).max())
print("rho impl", max(abs(np.linalg.eigvals(D))))
T=np.array([[-2,1,0],[1,-2,1],[0,1,-2.]])
for l in (0.4,4.0): print("rho expl",l, max(abs(np.linalg.eigvals(np.eye(3)+l*T))))
ev=np.linalg.eigvalsh(T); print("T eig", ev, ev/h**2, -np.pi**2)
