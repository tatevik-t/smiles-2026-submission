"""Distributional audit of hidden-state embeddings across the three splits and
against the unlabeled competition test set.

Answers two questions:
  1. Are dataset.csv's train/val/test splits IID?
  2. Is dataset.csv's distribution similar to data/test.csv (what we submit on)?

Metrics per pair (A, B):
  - centroid cosine similarity      [-1, 1], 1 = identical mean direction
  - centroid L2 distance            scale-dependent
  - domain-classifier AUROC (LR)    0.5 = indistinguishable, 1.0 = clear shift

Reads aggregation choice from AGG_STRATEGY env var (defaults to last_token,
matching solution.py defaults).

Run from smiles/repo/:
  python embeddings_audit.py
"""

from __future__ import annotations

import json
import os
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


def extract_features(texts: list[str], model, tokenizer, device) -> np.ndarray:
    """Run Qwen forward over texts, return aggregated features (N, D)."""
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


def centroid_metrics(A: np.ndarray, B: np.ndarray) -> dict:
    mA, mB = A.mean(axis=0), B.mean(axis=0)
    cos = float(np.dot(mA, mB) / (np.linalg.norm(mA) * np.linalg.norm(mB) + 1e-12))
    l2 = float(np.linalg.norm(mA - mB))
    return {"cos_sim": cos, "l2_dist": l2}


def domain_classifier_auroc(A: np.ndarray, B: np.ndarray, *, n_splits: int = 3, seed: int = 0) -> dict:
    """Train LR to distinguish A vs B with K-fold CV; report mean AUROC and acc.

    AUROC ~0.5 means the two distributions are statistically indistinguishable
    at the linear-classifier level; AUROC ~1.0 means trivially separable.
    """
    X = np.vstack([A, B])
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))]).astype(int)
    if len(A) < n_splits or len(B) < n_splits:
        return {"auroc_mean": float("nan"), "auroc_std": float("nan"), "n_a": len(A), "n_b": len(B)}
    scaler = StandardScaler()
    aurocs: list[float] = []
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        Xtr_s = scaler.fit_transform(X[tr])
        Xte_s = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xtr_s, y[tr])
        p = clf.predict_proba(Xte_s)[:, 1]
        aurocs.append(roc_auc_score(y[te], p))
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "n_a": len(A),
        "n_b": len(B),
    }


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
    print(f"[Extract] dataset.csv ({len(df)} samples) ...")
    X_dataset = extract_features(
        [f"{r['prompt']}{r['response']}" for _, r in df.iterrows()], model, tokenizer, device
    )
    print(f"[Extract] test.csv ({len(df_test)} samples) ...")
    X_testcsv = extract_features(
        [f"{r['prompt']}{r['response']}" for _, r in df_test.iterrows()], model, tokenizer, device
    )
    print(f"Done in {time.time() - t0:.1f}s.  dataset shape={X_dataset.shape}  test shape={X_testcsv.shape}")

    # Reproduce the same 70/15/15 stratified split as splitting.py SPLIT_STRATEGY=single.
    idx = np.arange(len(y_all))
    idx_tv, idx_test = train_test_split(idx, test_size=0.15, random_state=42, stratify=y_all)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.15 / 0.85, random_state=42, stratify=y_all[idx_tv]
    )

    pairs = {
        # Within dataset.csv splits.
        "train_vs_val":    (X_dataset[idx_train],          X_dataset[idx_val]),
        "train_vs_test":   (X_dataset[idx_train],          X_dataset[idx_test]),
        "val_vs_test":     (X_dataset[idx_val],            X_dataset[idx_test]),
        # Cross-dataset shift (the important one).
        "dataset_vs_testcsv":         (X_dataset,                  X_testcsv),
        # Per-class slices of the cross-dataset comparison.
        "hallucinated_dataset_vs_testcsv": (X_dataset[y_all == 1], X_testcsv),  # all test rows
        "truthful_dataset_vs_testcsv":     (X_dataset[y_all == 0], X_testcsv),
    }

    print("\n" + "=" * 86)
    print(f"  {'pair':<40} {'n_a':>5} {'n_b':>5} {'cos_sim':>8} {'l2_dist':>9} {'AUROC':>8}")
    print("-" * 86)

    report: dict = {"agg_strategy": AGG_STRATEGY, "pairs": {}}
    for name, (A, B) in pairs.items():
        cm = centroid_metrics(A, B)
        dc = domain_classifier_auroc(A, B)
        row = {**cm, **dc}
        report["pairs"][name] = row
        auroc_disp = f"{dc['auroc_mean']:.3f}±{dc['auroc_std']:.2f}"
        print(
            f"  {name:<40} {dc['n_a']:>5} {dc['n_b']:>5} "
            f"{cm['cos_sim']:>8.4f} {cm['l2_dist']:>9.3f} {auroc_disp:>8}"
        )
    print("=" * 86)
    print("  AUROC ~0.50: indistinguishable distributions.  "
          "AUROC >>0.70: clear shift.")

    # PCA: variance carried by the top components, on the combined embedding cloud.
    from sklearn.decomposition import PCA

    pca = PCA(n_components=10)
    pca.fit(np.vstack([X_dataset, X_testcsv]))
    cum = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nPCA cumulative variance (top 10 components): "
          + " ".join(f"{v:.2f}" for v in cum))
    report["pca_cum_variance_top10"] = cum.tolist()

    out_dir = Path("logs/audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "embeddings_audit.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
