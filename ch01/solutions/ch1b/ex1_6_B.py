"""Ex 1.6.B: Newton on double root f=(x-1)^2, plus modified Newton (m=2)."""
f  = lambda x: (x-1)**2
df = lambda x: 2*(x-1)
x = 2.0
print("plain Newton, x0=2")
for k in range(1, 61):
    if df(x) == 0: print(f"  f'(x)=0 at k={k}"); break
    xn = x - f(x)/df(x)
    e_old, e_new = abs(x-1), abs(xn-1)
    if k <= 5 or k % 10 == 0:
        print(f"  k={k:2d}: e={e_new:.3e}, e_k+1/e_k={e_new/e_old:.4f}")
    x = xn
    if e_new == 0: print(f"  exact at k={k}"); break
print(f"  after loop: k={k}, e={abs(x-1):.3e}")
# Same with generic double root f=(x-1)^2 (x+2), root 1 double
g  = lambda x: (x-1)**2*(x+2)
dg = lambda x: 2*(x-1)*(x+2) + (x-1)**2
x = 2.0
print("Newton on (x-1)^2(x+2), x0=2")
for k in range(1, 9):
    xn = x - g(x)/dg(x); print(f"  k={k}: e={abs(xn-1):.3e}, ratio={abs(xn-1)/abs(x-1):.4f}"); x = xn
# modified Newton x - m f/f'
x = 2.0
print("modified Newton m=2 on (x-1)^2(x+2)")
for k in range(1, 6):
    if dg(x) == 0: break
    xn = x - 2*g(x)/dg(x); print(f"  k={k}: e={abs(xn-1):.3e}"); x = xn
    if xn == 1: break
