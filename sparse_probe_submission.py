"""Produce a sparse-head-only predictions.csv for ensembling.

Pipeline:
  1. Load per-(layer, head) attention-to-prompt features for train (from
     head_attribution_features.npz).
  2. Forward Qwen on test.csv prompts to extract the SAME 336-dim per-head
     features for test samples.
  3. Train an XGBoost probe on the top-K heads (selected by full-train-set MI),
     then on a held-out val split tune the decision threshold.
  4. Predict on test.csv, write predictions.csv to logs/runs/sparse-head-K{K}/.

Run from smiles/repo/:
  python sparse_probe_submission.py            # K=100 (best from sparse_head_probe.py)
  python sparse_probe_submission.py --K 50     # try other K
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from model import MAX_LENGTH, _DEFAULT_MODEL, get_model_and_tokenizer

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"
BATCH_SIZE = 4
AUDIT_DIR = Path("logs/audits")
SPLIT_SEED = 42


def extract_head_features(rows, prompt_token_counts, tok, model, device):
    """Returns (N, n_layers * n_heads) numpy array of per-(layer,head)
    attention-to-prompt features at the response query positions."""
    N = len(rows)
    feats = None
    for start in tqdm(range(0, N, BATCH_SIZE), desc="head-feature extract", unit="batch"):
        end = min(start + BATCH_SIZE, N)
        batch_texts = [r["prompt"] + r["response"] for r in rows[start:end]]
        enc = tok(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc)
        attn_full = torch.stack(out.attentions, dim=0).float()
        if feats is None:
            n_layers = attn_full.shape[0]
            n_heads = attn_full.shape[2]
            feats = np.zeros((N, n_layers * n_heads), dtype=np.float32)
        attention_mask = enc["attention_mask"].cpu()
        for bi in range(end - start):
            mask = attention_mask[bi]
            last_pos = int(mask.nonzero(as_tuple=False)[-1].item())
            real_length = last_pos + 1
            ptc = prompt_token_counts[start + bi]
            if ptc <= 0 or ptc >= real_length:
                continue
            sa = attn_full[:, bi, :, ptc:real_length, :real_length]
            sa = sa / sa.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            prompt_mass = sa[..., :ptc].sum(dim=-1)
            feats[start + bi] = prompt_mass.mean(dim=-1).flatten().cpu().numpy()
    return feats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=100, help="number of top heads to use (K=100 was best in sparse_head_probe.py)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  K={args.K}")

    # Load train features from head_attribution.
    feats_path = AUDIT_DIR / "head_attribution_features.npz"
    if not feats_path.exists():
        raise SystemExit("Missing head_attribution_features.npz — run head_attribution.py first.")
    train_data = np.load(feats_path)
    X_train_full = train_data["feats"]                                   # (N_train, 336)
    y_train = train_data["y"]
    print(f"Loaded train head features: {X_train_full.shape}")

    # Load model with eager attention to get attention matrices for test.csv.
    print(f"[Model] loading '{_DEFAULT_MODEL}' (eager attn) ...")
    tok = get_model_and_tokenizer()[1]
    model = AutoModelForCausalLM.from_pretrained(
        _DEFAULT_MODEL,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()
    model.to(device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    df_test = pd.read_csv(TEST_FILE)
    test_rows = [{"prompt": r["prompt"], "response": r["response"]} for _, r in df_test.iterrows()]
    test_prompt_token_counts = [len(tok.encode(r["prompt"], add_special_tokens=False)) for r in test_rows]
    print(f"[Test] {len(test_rows)} samples")

    t0 = time.time()
    X_test_full = extract_head_features(test_rows, test_prompt_token_counts, tok, model, device)
    print(f"Test extraction done in {time.time() - t0:.1f}s.  shape: {X_test_full.shape}")

    # Top-K head selection by MI on FULL train set (not per-fold, since we're
    # producing a single submission). This has a mild leakage flavor but mirrors
    # how the final submission probe is fit on all 689 samples.
    print(f"\nComputing MI over {X_train_full.shape[1]} heads on full train set ...")
    mi = mutual_info_classif(X_train_full, y_train, random_state=42, n_neighbors=5)
    top_k_idx = np.argsort(-mi)[: args.K]
    print(f"Top-{args.K} heads selected; MI range {mi[top_k_idx[-1]]:.4f}..{mi[top_k_idx[0]]:.4f}")

    X_train_sel = X_train_full[:, top_k_idx]
    X_test_sel = X_test_full[:, top_k_idx]

    # Hold out 10% of train for threshold tuning (mirror solution.py final-probe).
    idx_ft, idx_fv = train_test_split(
        np.arange(len(y_train)), test_size=0.10, random_state=SPLIT_SEED, stratify=y_train,
    )
    scaler = StandardScaler()
    X_ft = scaler.fit_transform(X_train_sel[idx_ft])
    X_fv = scaler.transform(X_train_sel[idx_fv])
    X_te = scaler.transform(X_test_sel)

    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", n_jobs=-1,
    )
    clf.fit(X_ft, y_train[idx_ft])
    probs_val = clf.predict_proba(X_fv)[:, 1]
    cands = np.linspace(0.05, 0.95, 181)
    best_t, best_s = 0.5, -1.0
    for t in cands:
        preds = (probs_val >= t).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        s = accuracy_score(y_train[idx_fv], preds)
        if s > best_s:
            best_s = s
            best_t = float(t)
    print(f"Chosen threshold on val: {best_t:.4f}  val_acc={best_s*100:.2f}%")

    probs_test = clf.predict_proba(X_te)[:, 1]
    preds_test = (probs_test >= best_t).astype(int)
    print(f"Predictions distribution: {(preds_test == 1).sum()}/{len(preds_test)} hallucinated")

    out_dir = Path(f"logs/runs/sparse-head-K{args.K}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df_test.index, "label": preds_test}).to_csv(out_dir / "predictions.csv", index=False)
    print(f"Saved to {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
