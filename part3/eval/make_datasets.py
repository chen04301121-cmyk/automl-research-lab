"""Build the Part 3 evaluation datasets (written to part3/eval/data/, git-ignored).

Each file isolates ONE behaviour we want to test. Faults are injected with a fixed
seed so every system sees exactly the same data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
FRAUD = REPO / "datasets" / "airflow" / "fraud_clean_latest.csv"
TITANIC = REPO / "datasets" / "Titanic-Dataset.csv"
JUNK = ["987 note:11", "XDs=ed&3", "n/a?", "12,5O", "--"]   # junk values from the Week 11 slides


def signal_table(seed: int = 7) -> pd.DataFrame:
    """Imbalanced (~11% positive) table with real signal: model choice + threshold matter."""
    X, y = make_classification(n_samples=6000, n_features=12, n_informative=6, n_redundant=2,
                               weights=[0.9], flip_y=0.02, class_sep=1.0, random_state=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(1, 13)]).round(4)
    rng = np.random.default_rng(seed)
    df.insert(0, "account_id", np.arange(100000, 100000 + len(df)))
    df["segment"] = rng.choice(["retail", "smb", "corporate"], size=len(df))
    df["label"] = y
    return df


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if not FRAUD.is_file():
        raise SystemExit(f"{FRAUD} missing -> run the fraud_etl_pipeline DAG first")
    rng = np.random.default_rng(0)

    # 1-2: leakage (original name, then renamed so a hard-coded drop list cannot catch it)
    fraud = pd.read_csv(FRAUD)
    fraud.to_csv(DATA / "fraud_leaky.csv", index=False)
    fraud.rename(columns={"fraud_probability": "vendor_risk_score"}).to_csv(DATA / "fraud_leak_renamed.csv", index=False)
    # 9: no signal at all (leak removed) -> the right move is to stop early
    fraud.drop(columns=["fraud_probability"]).to_csv(DATA / "fraud_no_signal.csv", index=False)

    # 3: balanced, clean public dataset
    shutil.copy(TITANIC, DATA / "titanic.csv")

    # 4: clean imbalanced table with real signal
    base = signal_table()
    base.to_csv(DATA / "signal_clean.csv", index=False)

    # 5: numeric contamination in f3 (3% of rows)
    dirty = base.copy()
    dirty["f3"] = dirty["f3"].astype(object)
    idx = rng.choice(len(dirty), size=int(0.03 * len(dirty)), replace=False)
    dirty.loc[idx, "f3"] = rng.choice(JUNK, size=len(idx))
    dirty.to_csv(DATA / "signal_dirty_numeric.csv", index=False)

    # 6: 8% duplicated rows appended (a re-run that appended twice)
    dup_idx = rng.choice(len(base), size=int(0.08 * len(base)), replace=False)
    pd.concat([base, base.iloc[dup_idx]], ignore_index=True).to_csv(DATA / "signal_duplicates.csv", index=False)

    # 7: 1.5% invalid labels
    bad = base.copy()
    bad["label"] = bad["label"].astype(object)
    idx = rng.choice(len(bad), size=int(0.015 * len(bad)), replace=False)
    bad.loc[idx, "label"] = rng.choice(["unknown", "2", ""], size=len(idx))
    bad.to_csv(DATA / "signal_invalid_labels.csv", index=False)

    print("datasets written to", DATA)
    for p in sorted(DATA.glob("*.csv")):
        print(f"  {p.name:28s} {sum(1 for _ in open(p)) - 1:>6} rows")


if __name__ == "__main__":
    main()
