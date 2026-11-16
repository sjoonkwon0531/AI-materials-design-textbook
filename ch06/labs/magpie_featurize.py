"""Shared featurizer: Magpie ElementProperty features for expt_gap_matbench.csv.
Writes labs/magpie_expt_gap.parquet atomically (temp name then os.replace).
Safe to run if another author already created it (skips)."""
import os, time, warnings
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "magpie_expt_gap.parquet")

if os.path.exists(OUT):
    print("exists:", OUT)
else:
    t0 = time.time()
    df = pd.read_csv(os.path.join(HERE, "expt_gap_matbench.csv"))
    df["composition"] = [Composition(f) for f in df["formula"]]
    ep = ElementProperty.from_preset("magpie")
    ep.set_n_jobs(1)
    X = ep.featurize_dataframe(df, "composition", ignore_errors=True, pbar=False)
    X = X.drop(columns=["composition"])
    tmp = OUT + f".tmp{os.getpid()}"
    X.to_parquet(tmp, index=False)
    os.replace(tmp, OUT)
    print("wrote", OUT, X.shape, f"{time.time()-t0:.1f} s")
