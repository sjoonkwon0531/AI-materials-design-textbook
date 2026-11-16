# Ex 2.4.9: 2D FEM stiffness stencils: P1 on right-isosceles triangles (both diagonal choices),
#           Q1 bilinear squares, P1 on equilateral triangles. Compare with the FDM 5-point stencil.
import numpy as np
from collections import defaultdict
np.set_printoptions(precision=4, suppress=True)

def p1_elem(P):                       # P: 3x2 vertex coords -> 3x3 element stiffness
    M = np.c_[np.ones(3), P]; C = np.linalg.inv(M); grads = C[1:, :].T   # rows: grad phi_i
    area = 0.5*abs(np.linalg.det(M)); return area*grads@grads.T

def q1_elem(h):                       # bilinear square element via 2x2 Gauss
    K = np.zeros((4, 4)); g = [-1/np.sqrt(3), 1/np.sqrt(3)]
    sx = np.array([-1, 1, 1, -1]); sy = np.array([-1, -1, 1, 1])     # nodes (0,0),(h,0),(h,h),(0,h)
    for xi in g:
        for et in g:
            dN = np.array([sx*(1+sy*et)/4, sy*(1+sx*xi)/4])*(2/h)        # d/dx, d/dy
            K += dN.T@dN*(h/2)**2
    return K

def stencil_square_mesh(kind, h=1.0, n=4):
    """assemble on an n x n block of squares, return couplings of the centre node (n/2,n/2)."""
    S = defaultdict(float); c = (n//2, n//2)
    for i in range(n):
        for j in range(n):
            q = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
            if kind == 'Q1':
                elems = [(q, q1_elem(h))]
            else:
                tris = [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] if kind == 'P1 diag /' else [[q[0], q[1], q[3]], [q[1], q[2], q[3]]]
                elems = [(t, p1_elem(np.array(t, float)*h)) for t in tris]
            for nodes, K in elems:
                if c in nodes:
                    a = nodes.index(c)
                    for b, nb in enumerate(nodes): S[(nb[0]-c[0], nb[1]-c[1])] += K[a, b]
    return {k: round(v, 12) for k, v in S.items() if abs(v) > 1e-12}

for kind in ['P1 diag /', 'P1 diag \\', 'Q1']:
    S = stencil_square_mesh(kind)
    print(f"{kind:10s}: {len(S)}-point stencil ->", dict(sorted(S.items())))

# equilateral triangle mesh: the 6 triangles around a node
P = [np.array([np.cos(np.pi/3*m), np.sin(np.pi/3*m)]) for m in range(6)]
S = defaultdict(float)
for m in range(6):
    K = p1_elem(np.array([[0, 0], P[m], P[(m+1) % 6]]))
    S['centre'] += K[0, 0]; S[f'nbr{m}'] += K[0, 1]; S[f'nbr{(m+1)%6}'] += K[0, 2]
print("equilateral:", {k: round(v, 6) for k, v in S.items()}, " (1/sqrt3 =", round(1/np.sqrt(3), 6), ")")

# cot formula check: off-diagonal K_ij = -(cot a + cot b)/2 over the two opposite angles
print("right-isosceles hypotenuse edge: -(cot90+cot90)/2 =", -(0+0)/2, "; leg edge: -(cot45+cot45)/2 =", -(1+1)/2)

# global comparison on the unit square, -lap u = f with u = sin(pi x) sin(pi y): P1 (consistent load) vs FDM
def solve(n, method):
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    h = 1.0/n; idx = lambda i, j: (i-1)*(n-1)+(j-1); Nn = (n-1)**2
    f = lambda x, y: 2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
    A = lil_matrix((Nn, Nn)); b = np.zeros(Nn)
    for i in range(1, n):
        for j in range(1, n):
            r = idx(i, j); A[r, r] = 4
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if 0 < i+di < n and 0 < j+dj < n: A[r, idx(i+di, j+dj)] = -1
            x, y = i*h, j*h
            if method == 'FDM': b[r] = h*h*f(x, y)
            else:   # P1 '/' mesh, consistent load b_i = sum_e int f phi_i (3-pt edge-midpoint rule, exact for quadratics)
                tot = 0.0
                for (ax, ay), (bx, by) in [((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (-1, 0)), ((-1, 0), (-1, -1)), ((-1, -1), (0, -1)), ((0, -1), (1, 0))]:
                    V = np.array([[x, y], [x+ax*h, y+ay*h], [x+bx*h, y+by*h]]); area = h*h/2
                    mids = [(V[0]+V[1])/2, (V[1]+V[2])/2, (V[2]+V[0])/2]; phis = [0.5, 0.0, 0.5]
                    tot += area/3*sum(f(*m)*p for m, p in zip(mids, phis))
                b[r] = tot
    u = spsolve(A.tocsr(), b); xs = np.arange(1, n)*h
    ue = np.outer(np.sin(np.pi*xs), np.sin(np.pi*xs)).ravel()
    return np.abs(u-ue).max()
for n in [8, 16, 32]:
    print(f"n={n:3d}: nodal max err FDM={solve(n,'FDM'):.3e}   P1(consistent load)={solve(n,'P1'):.3e}")
