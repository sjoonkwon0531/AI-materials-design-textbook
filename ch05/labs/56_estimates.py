"""Order-of-magnitude numbers quoted in Secs. 5.5-5.6 (intro of 5.5 and Estimate Box 56_qc)."""
import numpy as np
NA = 6.02214076e23
n_Al = 2.70/26.98*NA*1e6            # atoms per m^3 (rho=2.70 g/cm^3, M=26.98 g/mol)
print(f"Al atom density            = {n_Al:.3e} m^-3")
print(f"atoms in (50 um)^3          = {n_Al*(50e-6)**3:.2e}")
N_tot = n_Al*(1e-6)**3
V_core = np.pi*(5e-9)**2*1e-6
N_core = n_Al*V_core
N_rep_out = (N_tot-N_core)/1e4
print(f"atoms in (1 um)^3           = {N_tot:.2e}")
print(f"core cylinder r=5nm,L=1um   : V = {V_core:.2e} m^3, atoms = {N_core:.2e}")
print(f"outer repatoms (1e4 atoms/element) = {N_rep_out:.2e};  total DOF ~ {N_core+N_rep_out:.2e};  reduction = {N_tot/(N_core+N_rep_out):.1e}")
