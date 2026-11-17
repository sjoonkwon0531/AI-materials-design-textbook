"""Export the Python labs' data tables and train/test indices for the Julia Labs.

Writes data/magpie_expt_gap.csv, data/67_features_clean.csv, data/xrd_peaks.csv, data/xrd_labels.csv
(Julia reads CSV; the Python Labs read the Parquet/NPZ originals) and data/splits/*.csv.

The Julia Labs in ch07/julia and ch08/julia read these files via read_idx() so that they use exactly the
same samples as the Python Labs (NumPy and Julia random streams differ).  Each file has one column `idx`
holding 0-based row indices; read_idx() adds 1.
Run once, from this folder:  python export_splits.py   (a few seconds)."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "splits")
os.makedirs(OUT, exist_ok=True)


def save(name, idx):
    pd.DataFrame({"idx": np.asarray(idx, dtype=np.int64)}).to_csv(os.path.join(OUT, name), index=False)


# Lab 7.3 / 7.4 (71_optim.py, 71_double_descent.py): Ch.6 fold 0 on the Magpie band-gap set
df = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))
X = df.drop(columns=["formula", "target"]).values
tr, te = next(iter(KFold(5, shuffle=True, random_state=0).split(X)))
tr_in, va = train_test_split(tr, test_size=0.1, random_state=0)
save("fold0_train.csv", tr); save("fold0_test.csv", te)
save("fold0_inner_train.csv", tr_in); save("fold0_val.csv", va)
save("dd_sub600.csv", np.random.default_rng(0).choice(tr, 600, replace=False))

# Lab 7.17 (75_lora_bandgap.py): source / target permutations and the target subsets
from pymatgen.core import Composition
is_t = np.array([bool({str(e) for e in Composition(f).elements} & {"N", "P", "As", "Sb"}) for f in df.formula])
rng = np.random.default_rng(0)
src = rng.permutation(np.where(~is_t)[0]); tgt = rng.permutation(np.where(is_t)[0])
save("lora_src_perm.csv", src); save("lora_tgt_perm.csv", tgt)
for N in (50, 200):
    for seed in range(5):
        save(f"lora_N{N}_seed{seed}.csv", np.random.default_rng(seed).choice(tgt[:400], N, replace=False))

# Lab 7.18 (76_mlp_bandgap.py): random and chemical-system-grouped 5-fold test indices
d6 = pd.read_parquet(os.path.join(HERE, "67_features_clean.parquet"))
feat = [c for c in d6.columns if c not in ("formula", "target", "chemsys", "reduced")]
X6 = d6[feat].values
for tag, splits in (("random", KFold(5, shuffle=True, random_state=0).split(X6)),
                    ("group", GroupKFold(5).split(X6, d6.target.values, d6.chemsys.values))):
    for f, (_, te6) in enumerate(splits):
        save(f"cv_{tag}_fold{f}.csv", te6)

# Lab 7.16 (74_pretrain_ethanol.py): fine-tuning subsets drawn from the 300 K pool (frames 0-1999)
for N in (50, 200):
    for seed in range(5):
        save(f"pretrain_N{N}_seed{seed}.csv", np.random.default_rng(100 + seed).choice(np.arange(2000), N, replace=False))

# Labs 7.21, 7.22, 8.10 (72b_utils.split_300K, 77_*.py, 83_mlip_sam.py): ethanol 300 K pool order
save("ethanol_pool_perm.csv", np.random.default_rng(0).permutation(2000))

# Data tables for ch7_common.jl (load_magpie, load_features149) and lab_7_5.jl
df.to_csv(os.path.join(HERE, "magpie_expt_gap.csv"), index=False)
d6.to_csv(os.path.join(HERE, "67_features_clean.csv"), index=False)
xrd = os.path.join(HERE, "..", "labs", "72a_xrd_peaks.npz")   # written by labs/72a_xrd_cnn.py on its first run
if os.path.exists(xrd):
    d = np.load(xrd, allow_pickle=True)
    pid = np.concatenate([np.full(len(x), i) for i, x in enumerate(d["x"])])
    pd.DataFrame({"pid": pid, "two_theta": np.concatenate(d["x"]), "intensity": np.concatenate(d["y"])}).to_csv(
        os.path.join(HERE, "xrd_peaks.csv"), index=False)
    pd.DataFrame({"pid": np.arange(len(d["lab"])), "label": d["lab"]}).to_csv(os.path.join(HERE, "xrd_labels.csv"), index=False)
else:
    print("skipped xrd_peaks.csv: run labs/72a_xrd_cnn.py first")

print("wrote", len(os.listdir(OUT)), "split files to", OUT, "and the data CSVs to", HERE)
