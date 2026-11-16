"""Ex 5.5.1: Voigt, Reuss, 2D Hashin-Shtrikman bounds for k1=1, k2=10, f2=0.3."""
k1, k2, f2 = 1.0, 10.0, 0.3; f1 = 1 - f2; d = 2
V = f1*k1 + f2*k2; R = 1/(f1/k1 + f2/k2)
HSl = k1 + f2/(1/(k2-k1) + f1/(d*k1)); HSu = k2 + f1/(1/(k1-k2) + f2/(d*k2))
print(f"Voigt={V:.4f}  Reuss={R:.4f}  HS-={HSl:.4f}  HS+={HSu:.4f}")
print(f"HS band width / Voigt-Reuss width = {(HSu-HSl)/(V-R):.3f}")
# laminate tensor: parallel = Voigt, normal = Reuss; isotropic invariants
print(f"laminate: k_par={V:.4f}, k_perp={R:.4f}, trace/2={(V+R)/2:.4f}, sqrt(det)={(V*R)**0.5:.4f}")
