"""Lab 6.7 step 1: featurize expt_gap (Magpie reused from shared file + Stoichiometry,
ValenceOrbital, IonProperty) and clean. Writes labs/67_features_clean.parquet."""
import os, time, warnings
import numpy as np, pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import Stoichiometry, ValenceOrbital, IonProperty
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
mag = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))   # shared Magpie file
print("raw entries:", len(mag), "| unique formula strings:", mag.formula.nunique())
comps = [Composition(f) for f in mag.formula]
extra = []
for F in (Stoichiometry(), ValenceOrbital(), IonProperty(fast=True)):
    rows = []
    for c in comps:
        try: rows.append(F.featurize(c))
        except Exception: rows.append([np.nan] * len(F.feature_labels()))
    extra.append(pd.DataFrame(rows, columns=F.feature_labels()))
df = pd.concat([mag.reset_index(drop=True)] + extra, axis=1)
feat = [c for c in df.columns if c not in ("formula", "target")]
print("features before cleaning:", len(feat))
# 1) NaN
nan_rows = df[feat].isna().any(axis=1)
print("rows with any NaN:", int(nan_rows.sum()))
df = df[~nan_rows].reset_index(drop=True)
# 2) constant / near-constant features
const = [c for c in feat if df[c].nunique() <= 1]
print("constant features dropped:", len(const), const[:4])
feat = [c for c in feat if c not in const]
mode_share = {c: df[c].value_counts(normalize=True).iloc[0] for c in feat}
near = sorted([c for c in feat if mode_share[c] >= 0.995], key=lambda c: -mode_share[c])
print("near-constant features (one value in >=99.5%% of rows): %d, e.g. %s (%.4f)" % (len(near), near[0], mode_share[near[0]]))
print("  -> kept for tree models; drop them (or fit the scaler per fold) before any linear model")
# 3) duplicates: same reduced composition
df["reduced"] = [Composition(f).reduced_formula for f in df.formula]
df["chemsys"] = ["-".join(sorted(e.symbol for e in Composition(f).elements)) for f in df.formula]
dup = df.duplicated("reduced", keep=False)
print("entries sharing a reduced formula:", int(dup.sum()), "in", df[dup].reduced.nunique(), "groups")
spread = df[dup].groupby("reduced").target.agg(lambda s: s.max() - s.min())
print("  max gap spread within such a group: %.2f eV" % spread.max())
df = df.groupby("reduced", as_index=False).agg({**{c: "first" for c in ["formula", "chemsys"] + feat},
                                                "target": "mean"})
print("after merging duplicates (target averaged):", len(df))
print("identical feature vectors remaining:", int(df.duplicated(feat).sum()))
print("chemical systems:", df.chemsys.nunique(), "| largest:", df.chemsys.value_counts().head(3).to_dict())
print("metals (gap=0): %d (%.1f%%)" % ((df.target == 0).sum(), 100 * (df.target == 0).mean()))
print("final features:", len(feat))
tmp = os.path.join(HERE, f"67_features_clean.parquet.tmp{os.getpid()}")
df.to_parquet(tmp, index=False); os.replace(tmp, os.path.join(HERE, "67_features_clean.parquet"))
print("wall %.1f s" % (time.time() - t0))
