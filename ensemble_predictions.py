"""Ensemble predictions.csv across multiple runs.

Averages per-id labels across N runs' predictions.csv files. Bayesian-style
majority vote with optional probability calibration.

Default: average across the five per-seed runs from phase 6
(logs/runs/seed-{0..4}-acc-thresh/predictions.csv), plus the canonical 5-fold
final-submission run.

Run from smiles/repo/:
  python ensemble_predictions.py
  python ensemble_predictions.py --runs seed-0-acc-thresh seed-1-acc-thresh ...
  python ensemble_predictions.py --out predictions.csv  # overwrite canonical
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RUNS = [
    "seed-0-acc-thresh",
    "seed-1-acc-thresh",
    "seed-2-acc-thresh",
    "seed-3-acc-thresh",
    "seed-4-acc-thresh",
    "final-submission",
]
RUNS_DIR = Path("logs/runs")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    p.add_argument("--out", default="logs/runs/ensemble/predictions.csv")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="vote threshold; 0.5 = majority")
    args = p.parse_args()

    dfs: list[pd.DataFrame] = []
    for run in args.runs:
        path = RUNS_DIR / run / "predictions.csv"
        if not path.exists():
            print(f"  ! skipping {run} (no predictions.csv)")
            continue
        df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
        dfs.append(df)
        print(f"  + {run}: {len(df)} predictions, {df['label'].mean():.2%} hallucinated")
    if not dfs:
        raise SystemExit("no runs to ensemble")

    # Verify the id columns align (defensive — shouldn't happen if runs share test.csv).
    base_ids = dfs[0]["id"].tolist()
    for i, df in enumerate(dfs[1:], 1):
        if df["id"].tolist() != base_ids:
            raise SystemExit(f"id mismatch between {args.runs[0]} and {args.runs[i]}")

    labels = np.stack([df["label"].values for df in dfs])  # (n_runs, n_predictions)
    avg = labels.mean(axis=0)
    ensemble = (avg >= args.threshold).astype(int)
    n_change_per_run = [(labels[i] != ensemble).sum() for i in range(len(labels))]

    print(f"\nEnsembled {len(dfs)} runs.")
    print(f"  ensemble class distribution: "
          f"{(ensemble == 1).sum()} hallucinated / {(ensemble == 0).sum()} truthful")
    print(f"  flips vs each input run: {n_change_per_run}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": base_ids, "label": ensemble}).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
