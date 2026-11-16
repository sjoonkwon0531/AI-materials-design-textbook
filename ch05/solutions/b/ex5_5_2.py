"""Ex 5.5.2: D2Q9 lattice moment identities, exact rational arithmetic (sympy)."""
import sympy as sp, itertools
c = [(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)]
w = [sp.Rational(4,9)] + [sp.Rational(1,9)]*4 + [sp.Rational(1,36)]*4
print("sum w =", sum(w))
print("sum w c_a =", [sum(wi*ci[a] for wi, ci in zip(w, c)) for a in range(2)])
print("sum w c_a c_b =", [[sum(wi*ci[a]*ci[b] for wi, ci in zip(w, c)) for b in range(2)] for a in range(2)])
print("max |sum w c_a c_b c_g| =", max(abs(sum(wi*ci[a]*ci[b]*ci[g] for wi, ci in zip(w, c))) for a, b, g in itertools.product(range(2), repeat=3)))
cs2 = sp.Rational(1,3)
ok = all(sum(wi*ci[a]*ci[b]*ci[g]*ci[d] for wi, ci in zip(w, c)) ==
         cs2**2*(int(a==b)*int(g==d)+int(a==g)*int(b==d)+int(a==d)*int(b==g)) for a,b,g,d in itertools.product(range(2), repeat=4))
print("4th-moment isotropy holds:", ok)
rho, ux, uy = sp.symbols('rho u_x u_y')
feq = [wi*rho*(1 + (ci[0]*ux+ci[1]*uy)/cs2 + (ci[0]*ux+ci[1]*uy)**2/(2*cs2**2) - (ux**2+uy**2)/(2*cs2)) for wi, ci in zip(w, c)]
print("sum feq =", sp.simplify(sum(feq)))
print("sum c feq =", [sp.simplify(sum(fi*ci[a] for fi, ci in zip(feq, c))) for a in range(2)])
print("sum c c feq (xx, xy) =", sp.expand(sum(fi*ci[0]*ci[0] for fi, ci in zip(feq, c))), ",", sp.expand(sum(fi*ci[0]*ci[1] for fi, ci in zip(feq, c))))
# D2Q4 counter-example (axis velocities only, w=1/4, cs^2=1/2): 4th moment anisotropic
c4 = c[1:5]; w4 = [sp.Rational(1,4)]*4
m4 = lambda a,b,g,d: sum(wi*ci[a]*ci[b]*ci[g]*ci[d] for wi, ci in zip(w4, c4))
print("D2Q4: sum w c_x^4 =", m4(0,0,0,0), " sum w c_x^2 c_y^2 =", m4(0,0,1,1), "(isotropy needs ratio 3)")
