# Ex 4.7.3: Delta G_H = Delta E_H + 0.24 eV and Sabatier ranking by |Delta G_H|
dE = {'M1':-0.50, 'M2':-0.33, 'M3':-0.10, 'M4':+0.20}
dG = {k: v+0.24 for k,v in dE.items()}
for k in dE: print(f"{k}: dE_H={dE[k]:+.2f} -> dG_H={dG[k]:+.2f} eV")
rank = sorted(dG, key=lambda k: abs(dG[k]))
print("HER activity ranking (|dG_H| ascending):", " > ".join(f"{k}({dG[k]:+.2f})" for k in rank))
print("closest to Pt (dG_H ~ -0.09 eV):", min(dG, key=lambda k: abs(dG[k]-(-0.09))))
