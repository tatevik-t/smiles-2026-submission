"""For each phase-6 split seed, measure how similar the in-evaluation test slice
(104 samples of dataset.csv) is to the competition test set (data/test.csv).

Hypothesis: the seed whose test slice is most similar to test.csv gives the
most leaderboard-predictive test_acc.

For each seed:
  - reproduce the stratified 70/15/15 single split that splitting.py would use
  - compute cosine sim / L2 / domain-classifier AUROC between
    X_dataset[idx_test] and X_testcsv
  - join with the per-seed test_acc loaded from logs/runs/seed-{seed}-*/results.json

Run from smiles/repo/:
  CUDA_VISIBLE_DEVICES=1 python slice_similarity.py
"""

from __future__ import annotations

import json
from pathlib import Path
import time

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
SEEDS = [0, 1, 2, 3, 4, 42]
RUN_SUFFIX = "-acc-thresh"
RUNS_DIR = Path("logs/runs")


def extract_features(texts: list[str], model, tokenizer, device) -> np.ndarray:
    features: list[torch.Tensor] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = torch.stack(out.hidden_states, dim=1).float()
        mask_cpu = attention_mask.cpu()
        for i in range(hidden.size(0)):
            features.append(
                aggregation_and_feature_extraction(hidden[i], mask_cpu[i], use_geometric=False).cpu()
            )
    return np.vstack([f.numpy() for f in features])


def domain_auroc(A: np.ndarray, B: np.ndarray, *, n_splits: int = 3, seed: int = 0) -> tuple[float, float]:
    X = np.vstack([A, B])
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))]).astype(int)
    aurocs = []
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        s = StandardScaler()
        Xtr = s.fit_transform(X[tr])
        Xte = s.transform(X[te])
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, y[tr])
        aurocs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aurocs)), float(np.std(aurocs))


def load_run_test_acc(seed: int) -> float | None:
    """Pick test_acc from logs/runs/seed-{seed}{SUFFIX}/results.json if present."""
    candidate = RUNS_DIR / f"seed-{seed}{RUN_SUFFIX}" / "results.json"
    if not candidate.exists():
        return None
    with candidate.open() as f:
        r = json.load(f)
    return float(r.get("avg_test_accuracy", float("nan")))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  AGG_STRATEGY={AGG_STRATEGY}")

    df = pd.read_csv(DATA_FILE)
    df_test = pd.read_csv(TEST_FILE)
    y_all = np.array([int(float(h)) for h in df["label"]])

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
    print(f"Extraction done in {time.time() - t0:.1f}s.")

    rows: list[dict] = []
    for seed in SEEDS:
        # Reproduce splitting.py's single-split stratified 70/15/15 with this seed.
        idx = np.arange(len(y_all))
        idx_tv, idx_test = train_test_split(idx, test_size=0.15, random_state=seed, stratify=y_all)
        slice_emb = X_dataset[idx_test]

        # Centroid metrics.
        mA, mB = slice_emb.mean(axis=0), X_testcsv.mean(axis=0)
        cos = float(np.dot(mA, mB) / (np.linalg.norm(mA) * np.linalg.norm(mB) + 1e-12))
        l2 = float(np.linalg.norm(mA - mB))

        # Domain-classifier AUROC.
        auroc_mean, auroc_std = domain_auroc(slice_emb, X_testcsv)

        # Per-seed measured test_acc from the phase-6 sweep results.
        meas_test_acc = load_run_test_acc(seed)

        rows.append(
            {
                "seed": seed,
                "n_slice": int(len(idx_test)),
                "cos_sim": cos,
                "l2_dist": l2,
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "test_acc": meas_test_acc,
            }
        )

    print("\n" + "=" * 78)
    print(f"  {'seed':>4} {'n':>4} {'cos_sim':>8} {'l2_dist':>9} {'auroc':>14} {'test_acc':>10}")
    print("-" * 78)
    for r in rows:
        ta = f"{r['test_acc'] * 100:.2f}%" if r["test_acc"] is not None and r["test_acc"] == r["test_acc"] else "—"
        au = f"{r['auroc_mean']:.3f}±{r['auroc_std']:.2f}"
        print(
            f"  {r['seed']:>4} {r['n_slice']:>4} "
            f"{r['cos_sim']:>8.4f} {r['l2_dist']:>9.3f} {au:>14} {ta:>10}"
        )
    print("=" * 78)
    print(" Lower AUROC = test slice more like test.csv = measurement more trustworthy.")

    # Bottom-line: weighted average of test_acc by similarity (higher cos sim = more weight).
    paired = [r for r in rows if r["test_acc"] is not None and r["test_acc"] == r["test_acc"]]
    if paired:
        accs = np.array([r["test_acc"] for r in paired])
        # Use inverse of (domain-AUROC distance from 0.5) as a "trust" weight.
        # If a slice is indistinguishable from test.csv (auroc=0.5), trust it fully.
        trust = np.array([1.0 / max(abs(r["auroc_mean"] - 0.5), 0.01) for r in paired])
        weighted_acc = float(np.average(accs, weights=trust))
        plain_mean = float(np.mean(accs))
        plain_std = float(np.std(accs))
        print(f"\n  plain mean test_acc      : {plain_mean * 100:.2f}% ± {plain_std * 100:.2f}%")
        print(f"  trust-weighted test_acc  : {weighted_acc * 100:.2f}%")

    out_path = Path("logs/audits/slice_similarity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {"agg_strategy": AGG_STRATEGY, "rows": rows, "run_suffix": RUN_SUFFIX},
            f,
            indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
