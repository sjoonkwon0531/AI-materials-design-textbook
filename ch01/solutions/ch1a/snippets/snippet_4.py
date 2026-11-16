import numpy as np, networkx as nx
for G in [nx.erdos_renyi_graph(50, 0.1, seed=3),
          nx.stochastic_block_model([25, 25], [[0.3, 0.02], [0.02, 0.3]], seed=3)]:
    L = nx.laplacian_matrix(G, nodelist=range(50)).toarray().astype(float)
    w, U = np.linalg.eigh(L)
    emb = U[:, 1:3]                     # u2, u3 (u1은 상수 벡터)
    print(np.round(w[:5], 3), w[2]/w[1], np.bincount((U[:, 1] > 0).astype(int)))
