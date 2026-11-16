"""Ex 10.1.4: break-even when the 400 labels are 64-atom cells instead of 8-atom cells; production N=216."""
import numpy as np, json
r = json.load(open('../data/101_cost.json'))
t8 = r['dft']['8']['cpu']; pd = r['summary']['slope_dft']; cml = r['mlip']['216']
Ctr = 100.0
for tag, p in [('slope', pd), ('cubic', 3.0)]:
    c64 = t8 * 8 ** p; c216 = t8 * 27 ** p
    b8 = 400 * t8 + Ctr; b64 = 400 * c64 + Ctr
    print(f"[{tag} model] c_DFT(64) = {c64:.4g} s, c_DFT(216) = {c216:.4g} s, c_ML(216) = {cml:.4g} s")
    print(f"    8-atom labels : build {b8:.4g} s ({b8/3600:.3g} h)  S* = {b8/(c216-cml):.4g} steps")
    print(f"    64-atom labels: build {b64:.4g} s ({b64/3600:.4g} h = {b64/86400:.4g} d)  S* = {b64/(c216-cml):.4g} steps;"
          f" ratio of build costs {b64/b8:.4g}")
