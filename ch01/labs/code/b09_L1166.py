ATA = A_mat.T @ A_mat
cond = np.linalg.cond(ATA)
print(f"κ(A^T A) = {cond:.2f}")
# 출력: κ(A^T A) ≈ 5470