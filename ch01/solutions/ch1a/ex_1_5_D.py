# 연습문제 1.5.D: 50-노드 random graph의 Laplacian 스펙트럼과 spectral embedding
import numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def analyze(G, name, truth=None):
    Lm = nx.laplacian_matrix(G, nodelist=range(50)).toarray().astype(float)
    w, U = np.linalg.eigh(Lm)
    print(f"[{name}] connected={nx.is_connected(G)}, edges={G.number_of_edges()}")
    print(f"  smallest 5 eig: {np.round(w[:5], 4)}, largest: {w[-1]:.3f}")
    print(f"  u1 is constant? std(u1)={U[:,0].std():.1e}  (u1 = ±1/sqrt(50) = {1/np.sqrt(50):.4f})")
    print(f"  gap ratio lambda3/lambda2 = {w[2]/w[1]:.3f}")
    if truth is not None:
        lab = (U[:, 1] > 0).astype(int); acc = max(np.mean(lab == truth), np.mean(lab != truth))
        print(f"  Fiedler-sign partition accuracy vs planted blocks = {acc:.3f}")
        cut = sum(1 for a, b in G.edges if lab[a] != lab[b]); print(f"  edges cut by Fiedler partition = {cut}")
    else:
        lab = (U[:, 1] > 0).astype(int)
        cut = sum(1 for a, b in G.edges if lab[a] != lab[b]); print(f"  Fiedler partition sizes {np.bincount(lab)}, edges cut = {cut} of {G.number_of_edges()} ({cut/G.number_of_edges():.2f})")
    return U
G1 = nx.erdos_renyi_graph(50, 0.1, seed=3)
U1 = analyze(G1, "Erdos-Renyi G(50,0.1)")
G2 = nx.stochastic_block_model([25, 25], [[0.3, 0.02], [0.02, 0.3]], seed=3)
truth = np.array([0]*25 + [1]*25)
U2 = analyze(G2, "SBM 2x25 p_in=0.3 p_out=0.02", truth)
# 문자 그대로 '가장 작은 두 고유벡터'(u1,u2)로 배치하면: x좌표가 상수 → 1차원 직선
print("literal (u1,u2) embedding: x-range of u1 =", np.ptp(U2[:, 0]))
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].scatter(U2[:, 0], U2[:, 1], c=truth, cmap="coolwarm"); ax[0].set_title("SBM: (u1,u2) literal")
ax[1].scatter(U2[:, 1], U2[:, 2], c=truth, cmap="coolwarm"); ax[1].set_title("SBM: (u2,u3)")
ax[2].scatter(U1[:, 1], U1[:, 2], c="gray"); ax[2].set_title("ER: (u2,u3)")
fig.tight_layout(); fig.savefig("ex_1_5_D.png", dpi=120)
