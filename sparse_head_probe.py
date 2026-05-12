"""Sparse-probe analysis: can we hit the same accuracy with just the top-K
attention heads as with all 336?

Method:
  1. Load the per-(layer, head) attention-to-prompt features computed by
     head_attribution.py (shape: (689, 336)).
  2. For each fold of 5-fold stratified CV:
       a. On the train fold, compute mutual information per head with the
          training labels (no leakage onto the test fold).
       b. Select top-K heads by MI.
       c. Train a HallucinationProbe (XGBoost) on just those K features.
       d. Evaluate on the held-out test fold.
  3. Report mean test_acc / test_AUROC across folds for several K values.

If a 30-head sparse probe matches the 336-head dense probe (within noise),
the hallucination signal is highly localized — a strong mechanistic finding.

Run from smiles/repo/:
  python sparse_head_probe.py
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

OUT_DIR = Path("logs/audits")
SPLIT_SEED = 42
K_VALUES = [5, 10, 30, 50, 100, 336]


def fit_xgb(X, y, X_val=None, y_val=None):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def tune_threshold_for_accuracy(probs, y, lo=0.05, hi=0.95, n=181):
    """Pick threshold that maximizes accuracy on (probs, y) without going
    degenerate (predicting all one class)."""
    cands = np.linspace(lo, hi, n)
    best_t, best_score = 0.5, -1.0
    for t in cands:
        preds = (probs >= t).astype(int)
        n_pos = preds.sum()
        if n_pos == 0 or n_pos == len(preds):
            continue
        s = accuracy_score(y, preds)
        if s > best_score:
            best_score = s
            best_t = float(t)
    return best_t


def main() -> None:
    feats_path = OUT_DIR / "head_attribution_features.npz"
    if not feats_path.exists():
        raise SystemExit(f"Missing {feats_path}. Run head_attribution.py first.")
    data = np.load(feats_path)
    X_all = data["feats"]                                            # (N, 336)
    y_all = data["y"]
    n_layers, n_heads = int(data["n_layers"]), int(data["n_heads"])
    N, D = X_all.shape
    print(f"Loaded head features: X={X_all.shape}  y mean={y_all.mean():.3f}  ({n_layers}×{n_heads}={D} heads)")

    # Train/val carve for the within-fold validation (mirrors splitting.py 5-fold).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    folds = list(skf.split(np.arange(N), y_all))

    results: dict = {}
    head_counter: dict[int, Counter] = {k: Counter() for k in K_VALUES}

    for k in K_VALUES:
        accs, aurocs, thresholds = [], [], []
        for fold_i, (tr_idx, te_idx) in enumerate(folds):
            # Carve val from train portion for threshold tuning.
            tr2_idx, va_idx = train_test_split(
                tr_idx, test_size=0.15 / 0.8, random_state=SPLIT_SEED, stratify=y_all[tr_idx]
            )
            # MI computed on the FOLD's train portion only (no leakage).
            mi = mutual_info_classif(X_all[tr2_idx], y_all[tr2_idx], random_state=42, n_neighbors=5)
            top_k_idx = np.argsort(-mi)[:k]
            head_counter[k].update(top_k_idx.tolist())

            X_tr = X_all[tr2_idx][:, top_k_idx]
            X_va = X_all[va_idx][:, top_k_idx]
            X_te = X_all[te_idx][:, top_k_idx]

            clf = fit_xgb(X_tr, y_all[tr2_idx])
            probs_va = clf.predict_proba(X_va)[:, 1]
            t = tune_threshold_for_accuracy(probs_va, y_all[va_idx])
            probs_te = clf.predict_proba(X_te)[:, 1]
            preds_te = (probs_te >= t).astype(int)
            accs.append(accuracy_score(y_all[te_idx], preds_te))
            try:
                aurocs.append(roc_auc_score(y_all[te_idx], probs_te))
            except ValueError:
                aurocs.append(float("nan"))
            thresholds.append(t)

        results[k] = {
            "test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
            "test_auroc_mean": float(np.nanmean(aurocs)),
            "test_auroc_std": float(np.nanstd(aurocs)),
            "thresholds": [float(t) for t in thresholds],
        }

    print("\n" + "=" * 78)
    print(f"  {'K (heads)':>10} {'test_acc':>15} {'test_AUROC':>15} {'mean threshold':>15}")
    print("  " + "-" * 60)
    for k in K_VALUES:
        r = results[k]
        thresh_mean = float(np.mean(r["thresholds"]))
        print(
            f"  {k:>10} "
            f"{r['test_acc_mean']*100:>9.2f}% ± {r['test_acc_std']*100:>3.1f}pt"
            f"   {r['test_auroc_mean']*100:>9.2f}% ± {r['test_auroc_std']*100:>3.1f}pt"
            f"   {thresh_mean:>10.3f}"
        )
    print("=" * 78)
    print()
    print(f"  Reference: xgb-manifold-perhead-all-5fold (all 336 heads + manifold + ppl + heur, 1330 features):")
    print(f"    test_acc=74.02%  test_auroc=77.52%")
    print()
    print(f"  Reference: just-perhead xgb-perhead-attn-5fold (336 features only):")
    print(f"    test_acc=74.02%  test_auroc=76.34%")
    print()
    print("  Per-K: most-frequently-selected heads across folds (by layer):")
    for k in K_VALUES:
        top_5_global = head_counter[k].most_common(5)
        layers = [(idx // n_heads, idx % n_heads, cnt) for idx, cnt in top_5_global]
        compact = " ".join(f"L{l}H{h}({c})" for l, h, c in layers)
        print(f"    K={k:>3}: {compact}")

    with (OUT_DIR / "sparse_head_probe.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUT_DIR / 'sparse_head_probe.json'}")


if __name__ == "__main__":
    main()
