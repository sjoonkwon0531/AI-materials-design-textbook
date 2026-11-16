"""Export the Python labs' train/test indices for the Julia Labs (writes data/splits/*.csv).

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

print("wrote", len(os.listdir(OUT)), "files to", OUT)
