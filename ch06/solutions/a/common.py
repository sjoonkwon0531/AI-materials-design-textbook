import os, numpy as np, pandas as pd
from pymatgen.core import Composition
LABS="../ch06/labs"
def load_nonmetal():
    d = pd.read_parquet(os.path.join(LABS, "magpie_expt_gap.parquet"))
    d = d[d.target > 0].reset_index(drop=True)
    feats = [c for c in d.columns if c.startswith("MagpieData")]
    X = d[feats].to_numpy(float); y = d.target.to_numpy(float)
    keep = X.std(0) > 0
    X = X[:, keep]; names = [f.replace("MagpieData ", "") for f, k in zip(feats, keep) if k]
    frac_mode = np.array([pd.Series(X[:, j]).value_counts(normalize=True).iloc[0] for j in range(X.shape[1])])
    qc = frac_mode > 0.99
    Xf = X[:, ~qc]; namesf = list(np.array(names)[~qc])
    chemsys = np.array(["-".join(sorted(e.symbol for e in Composition(f).elements)) for f in d.formula])
    nel = np.array([len(Composition(f).elements) for f in d.formula])
    return d, Xf, y, namesf, chemsys, nel
