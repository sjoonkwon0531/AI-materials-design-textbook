"""Ex 10.2.4: r_jk for two bond angles, and the |v_i|^2 difference in units of w(2.35)^2."""
import numpy as np
b = 2.35
for th in [109.5, 100.0]:
    print(f"theta {th}: r_jk = {2*b*np.sin(np.radians(th)/2):.4f} A, cos = {np.cos(np.radians(th)):.5f}")
d = 2 * (np.cos(np.radians(109.5)) - np.cos(np.radians(100.0)))
print(f"|v|^2(109.5) - |v|^2(100) = {d:.4f} w(2.35)^2 ;  |v|^2 = w^2 (2 + 2cos): {2+2*np.cos(np.radians(109.5)):.4f} vs {2+2*np.cos(np.radians(100)):.4f}")
