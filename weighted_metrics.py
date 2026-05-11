"""For every (single-split or 5-fold) run, report the trust-weighted estimate of
leaderboard accuracy/AUROC — weighting each fold's measured test_acc by the
inverse domain-classifier-AUROC distance from 0.5 between that fold's test
slice and data/test.csv (closer to 0.5 = more like real test.csv = more trust).

For single-split runs, the trust weight applies to the single fold; the
weighted estimate equals the measured estimate (no averaging) but the trust
weight itself tells us how leaderboard-predictive the run's measurement was.

For 5-fold runs, the per-fold weights blend the five measurements into one
trust-weighted estimate, prioritising folds whose test slices best resemble
data/test.csv distributionally.

Run from smiles/repo/:
  CUDA_VISIBLE_DEVICES=1 python weighted_metrics.py --runs <run1> <run2> ...
  (omit --runs to analyze every results.json in logs/runs/)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from aggregation import AGG_STRATEGY, aggregation_and_feature_extraction
from model import MAX_LENGTH, get_model_and_tokenizer

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"
BATCH_SIZE = 8
RUNS_DIR = Path("logs/runs")


def extract_features(texts, model, tokenizer, device):
    feats = []
    for s in range(0, len(texts), BATCH_SIZE):
        batch = texts[s : s + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=am)
        h = torch.stack(out.hidden_states, dim=1).float()
        m = am.cpu()
        for i in range(h.size(0)):
            feats.append(aggregation_and_feature_extraction(h[i], m[i], use_geometric=False).cpu())
    return np.vstack([f.numpy() for f in feats])


def domain_auroc(A, B, *, n_splits=3, seed=0):
    X = np.vstack([A, B])
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))]).astype(int)
    aurocs = []
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        s = StandardScaler()
        Xtr = s.fit_transform(X[tr])
        Xte = s.transform(X[te])
        clf = LogisticRegression(max_iter=1000).fit(Xtr, y[tr])
        aurocs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aurocs))


def trust_weight(auroc: float, eps: float = 0.01) -> float:
    """Higher when domain-classifier AUROC is closer to 0.5 (more like test.csv)."""
    return 1.0 / max(abs(auroc - 0.5), eps)


def reproduce_split(y, strategy, seed):
    """Returns list of (train, val, test) like splitting.split_data, deterministic."""
    idx = np.arange(len(y))
    if strategy == "single":
        idx_tv, idx_te = train_test_split(idx, test_size=0.15, random_state=seed, stratify=y)
        idx_tr, idx_va = train_test_split(idx_tv, test_size=0.15 / 0.85, random_state=seed, stratify=y[idx_tv])
        return [(idx_tr, idx_va, idx_te)]
    if strategy == "5fold":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        out = []
        for tr_idx, te_idx in skf.split(idx, y):
            idx_tr, idx_va = train_test_split(tr_idx, test_size=0.15 / 0.8, random_state=seed, stratify=y[tr_idx])
            out.append((idx_tr, idx_va, te_idx))
        return out
    raise ValueError(f"unknown strategy {strategy!r}")


def infer_run_config(run_name: str) -> tuple[str, int]:
    """Best-effort: map run name → (split_strategy, split_seed). Uses naming conventions."""
    n = run_name
    if "5fold" in n:
        return ("5fold", 42)
    if n.startswith("seed-") and n.split("-")[1].isdigit():
        return ("single", int(n.split("-")[1]))
    # Default: most runs were single-split with seed=42.
    return ("single", 42)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=None,
                   help="run names under logs/runs/; default = all with results.json")
    args = p.parse_args()

    if args.runs:
        run_paths = [RUNS_DIR / r / "results.json" for r in args.runs]
    else:
        run_paths = sorted(RUNS_DIR.glob("*/results.json"))

    if not run_paths:
        raise SystemExit("no runs to analyze")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA_FILE)
    df_test = pd.read_csv(TEST_FILE)
    y_all = np.array([int(float(h)) for h in df["label"]])

    print(f"Device: {device}  AGG_STRATEGY={AGG_STRATEGY}")
    print("[Model] loading ...")
    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    t0 = time.time()
    X_dataset = extract_features(
        [f"{r['prompt']}{r['response']}" for _, r in df.iterrows()], model, tokenizer, device
    )
    X_testcsv = extract_features(
        [f"{r['prompt']}{r['response']}" for _, r in df_test.iterrows()], model, tokenizer, device
    )
    print(f"Extraction done in {time.time() - t0:.1f}s.\n")

    print("=" * 100)
    print(f"  {'run':<30} {'strategy':<7} {'seed':>4} "
          f"{'meas_acc':>9} {'norm_acc':>9} {'meas_auroc':>11} {'norm_auroc':>11} {'mean_trust':>10}")
    print("-" * 100)

    rows: list[dict] = []
    for results_path in run_paths:
        run_name = results_path.parent.name
        with results_path.open() as f:
            r = json.load(f)

        strategy, seed = infer_run_config(run_name)
        try:
            splits = reproduce_split(y_all, strategy, seed)
        except ValueError:
            continue

        fold_results = r.get("folds", [])
        if len(fold_results) != len(splits):
            # Stale results or non-matching strategy inference; skip but flag.
            continue

        per_fold_acc = []
        per_fold_auroc = []
        per_fold_trust = []
        for (_, _, idx_te), fold_r in zip(splits, fold_results):
            slice_emb = X_dataset[idx_te]
            d_auroc = domain_auroc(slice_emb, X_testcsv)
            t = trust_weight(d_auroc)
            per_fold_acc.append(fold_r.get("test_accuracy", float("nan")))
            per_fold_auroc.append(fold_r.get("test_auroc", float("nan")))
            per_fold_trust.append(t)

        per_fold_acc = np.array(per_fold_acc)
        per_fold_auroc = np.array(per_fold_auroc)
        per_fold_trust = np.array(per_fold_trust)

        meas_acc = float(np.nanmean(per_fold_acc))
        meas_auroc = float(np.nanmean(per_fold_auroc))
        valid = ~np.isnan(per_fold_acc)
        if valid.any():
            norm_acc = float(np.average(per_fold_acc[valid], weights=per_fold_trust[valid]))
            norm_auroc_arr = per_fold_auroc[valid]
            norm_trust_arr = per_fold_trust[valid]
            valid_auroc = ~np.isnan(norm_auroc_arr)
            norm_auroc = float(np.average(norm_auroc_arr[valid_auroc], weights=norm_trust_arr[valid_auroc])) \
                         if valid_auroc.any() else float("nan")
        else:
            norm_acc = norm_auroc = float("nan")
        mean_trust = float(per_fold_trust.mean())

        rows.append({
            "run": run_name,
            "strategy": strategy,
            "seed": seed,
            "meas_test_acc": meas_acc,
            "norm_test_acc": norm_acc,
            "meas_test_auroc": meas_auroc,
            "norm_test_auroc": norm_auroc,
            "mean_trust": mean_trust,
            "per_fold_trust": per_fold_trust.tolist(),
        })

    rows.sort(key=lambda r: -r["norm_test_acc"] if not np.isnan(r["norm_test_acc"]) else -1.0)

    for r in rows:
        print(f"  {r['run']:<30} {r['strategy']:<7} {r['seed']:>4} "
              f"{r['meas_test_acc']*100:>8.2f}% {r['norm_test_acc']*100:>8.2f}% "
              f"{r['meas_test_auroc']*100:>10.2f}% {r['norm_test_auroc']*100:>10.2f}% "
              f"{r['mean_trust']:>10.2f}")
    print("=" * 100)
    print("  norm_*: trust-weighted (weighted by 1 / |fold_domain_AUROC − 0.5|, capped).")
    print("  High mean_trust = test slices closely resemble data/test.csv distributionally.")

    out = Path("logs/audits/weighted_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
