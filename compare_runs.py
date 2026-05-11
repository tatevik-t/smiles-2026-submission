"""Tabulate per-run results.json files under logs/runs/ for cross-run comparison.

Invoke from smiles/repo/:
    python compare_runs.py
    python compare_runs.py --sort test_auroc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "logs" / "runs"

COLUMNS = [
    ("run", 26),
    ("folds", 5),
    ("feat_dim", 8),
    ("base_acc", 9),
    ("train_auroc", 11),
    ("val_acc", 8),
    ("val_f1", 8),
    ("val_auroc", 9),
    ("test_acc", 9),
    ("test_f1", 8),
    ("test_auroc", 10),
    ("extract_s", 9),
]


def _mean(folds: list[dict], key: str) -> float | None:
    """Mean of `key` across folds, ignoring NaN. Returns None if all NaN/missing."""
    import math
    vals = [f[key] for f in folds if key in f and not math.isnan(float(f[key]))]
    return float(sum(vals) / len(vals)) if vals else None


def fmt_pct(v: float | None) -> str:
    if v is None or v != v:  # NaN check
        return "—"
    return f"{v * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sort",
        default="test_auroc",
        choices=["test_auroc", "test_acc", "test_f1", "val_auroc"],
        help="column to sort by (desc)",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for results_path in sorted(RUNS_DIR.glob("*/results.json")):
        with results_path.open() as f:
            r = json.load(f)
        folds = r.get("folds", [])
        rows.append(
            {
                "run": results_path.parent.name,
                "folds": r.get("n_folds", 0),
                "feat_dim": r.get("feature_dim", 0),
                "base_acc": r.get("avg_baseline_accuracy"),
                "train_auroc": r.get("avg_train_auroc"),
                "val_acc": _mean(folds, "val_accuracy"),
                "val_f1": _mean(folds, "val_f1"),
                "val_auroc": r.get("avg_val_auroc"),
                "test_acc": r.get("avg_test_accuracy"),
                "test_f1": r.get("avg_test_f1"),
                "test_auroc": r.get("avg_test_auroc"),
                "extract_s": r.get("extract_time_s"),
            }
        )

    if not rows:
        print(f"No runs found under {RUNS_DIR}")
        return

    rows.sort(key=lambda r: (r[args.sort] if r[args.sort] == r[args.sort] else -1.0), reverse=True)

    header = "  ".join(f"{name:<{w}}" for name, w in COLUMNS)
    print(header)
    print("-" * len(header))

    for r in rows:
        cells = []
        for name, w in COLUMNS:
            v = r[name]
            if name == "run":
                cells.append(f"{v:<{w}}")
            elif name in ("folds", "feat_dim"):
                cells.append(f"{v:<{w}}")
            elif name == "extract_s":
                cells.append(f"{v:.1f}s".ljust(w) if v is not None else "—".ljust(w))
            else:
                cells.append(fmt_pct(v).ljust(w))
        print("  ".join(cells))


if __name__ == "__main__":
    main()
