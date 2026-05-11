"""Per-sample error analysis on the winning 5-fold canonical config.

Pipeline:
  1. Extract features for dataset.csv (last_token, final layer — winning config).
  2. Apply the same 5-fold split as splitting.py SPLIT_STRATEGY=5fold.
  3. Train HallucinationProbe per fold, predict on its test slice.
  4. Concatenate per-fold predictions so each of the 689 samples has exactly
     one prediction (the one made by the probe that didn't see it in training).
  5. Identify misclassifications, dump them with their prompt+response text
     plus token-length stats so we can spot systematic patterns.

Run from smiles/repo/:
  python error_analysis.py
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aggregation import AGG_STRATEGY, aggregation_and_feature_extraction
from model import MAX_LENGTH, get_model_and_tokenizer
from probe import HallucinationProbe
from splitting import split_data

DATA_FILE = "./data/dataset.csv"
BATCH_SIZE = 4
SEED = 42
OUT_DIR = Path("logs/audits")


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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  AGG_STRATEGY={AGG_STRATEGY}")

    df = pd.read_csv(DATA_FILE)
    y = np.array([int(float(h)) for h in df["label"]])
    texts = [f"{r['prompt']}{r['response']}" for _, r in df.iterrows()]
    response_texts = df["response"].tolist()
    prompt_texts = df["prompt"].tolist()
    n = len(y)
    print(f"Loaded {n} samples ({int(y.sum())} hallucinated / {n - int(y.sum()):d} truthful)")

    print("[Model] loading ...")
    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    t0 = time.time()
    X = extract_features(texts, model, tokenizer, device)
    print(f"Extracted features {X.shape} in {time.time() - t0:.1f}s.")

    import os
    os.environ["SPLIT_STRATEGY"] = "5fold"
    os.environ["SPLIT_SEED"] = str(SEED)
    import importlib, splitting
    importlib.reload(splitting)
    splits = splitting.split_data(y, df)

    # Per-sample arrays (filled by the fold whose test slice contains the sample).
    pred = np.full(n, -1, dtype=int)
    prob = np.full(n, np.nan, dtype=float)

    for fold_i, (idx_tr, idx_va, idx_te) in enumerate(splits):
        probe = HallucinationProbe()
        probe.fit(X[idx_tr], y[idx_tr])
        if idx_va is not None:
            probe.fit_hyperparameters(X[idx_va], y[idx_va])
        pred[idx_te] = probe.predict(X[idx_te])
        prob[idx_te] = probe.predict_proba(X[idx_te])[:, 1]
        acc = (pred[idx_te] == y[idx_te]).mean()
        print(f"  fold {fold_i + 1}: test_acc={acc:.4f}  threshold={probe._threshold:.4f}")

    # Now every sample has exactly one out-of-fold prediction.
    correct_mask = (pred == y)
    err_mask = ~correct_mask
    print(f"\nOverall out-of-fold accuracy: {correct_mask.mean():.4f}  "
          f"({correct_mask.sum()}/{n} correct, {err_mask.sum()} errors)")

    # Basic length statistics — first systematic-pattern hypothesis to check.
    response_lens = np.array([len(t) for t in response_texts])
    prompt_lens = np.array([len(t) for t in prompt_texts])
    print(f"\nResponse char length:  correct {response_lens[correct_mask].mean():.0f} "
          f"vs errors {response_lens[err_mask].mean():.0f}")
    print(f"Prompt char length:    correct {prompt_lens[correct_mask].mean():.0f} "
          f"vs errors {prompt_lens[err_mask].mean():.0f}")

    # Errors split by true class.
    err_h = err_mask & (y == 1)  # hallucinated misclassified as truthful (false negative)
    err_t = err_mask & (y == 0)  # truthful misclassified as hallucinated (false positive)
    print(f"\nFalse negatives (hallucinated→predicted truthful): {err_h.sum()}")
    print(f"False positives (truthful→predicted hallucinated):  {err_t.sum()}")

    # Confidence on errors: are they low-confidence flips or confident-wrong?
    conf_on_err = np.abs(prob[err_mask] - 0.5)
    if len(conf_on_err) > 0:
        print(f"\nProbe confidence on errors (|p - 0.5|): "
              f"mean={conf_on_err.mean():.3f}  median={np.median(conf_on_err):.3f}")
        print(f"Errors at |p - 0.5| < 0.10 (uncertain): {(conf_on_err < 0.10).sum()}")
        print(f"Errors at |p - 0.5| >= 0.30 (confidently wrong): {(conf_on_err >= 0.30).sum()}")

    # Hedge-word presence (manual heuristic — abstention markers).
    hedges = ("approximately", "i think", "i believe", "may have", "might be",
              "possibly", "perhaps", "around", "about ", "i'm not sure", "unclear")
    lc_responses = [t.lower() for t in response_texts]
    has_hedge = np.array([any(h in t for h in hedges) for t in lc_responses])
    print(f"\nResponses containing hedge words:")
    print(f"  hallucinated samples: {(has_hedge & (y == 1)).sum()} / {(y == 1).sum()}")
    print(f"  truthful samples:     {(has_hedge & (y == 0)).sum()} / {(y == 0).sum()}")

    # Dump the most confident-wrong samples for manual inspection.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    err_records = []
    for i in np.where(err_mask)[0]:
        err_records.append({
            "id": int(i),
            "true_label": int(y[i]),
            "pred_label": int(pred[i]),
            "prob_hallucinated": float(prob[i]),
            "confidence": float(abs(prob[i] - 0.5)),
            "response_len": int(response_lens[i]),
            "prompt_len": int(prompt_lens[i]),
            "response": response_texts[i],
            "prompt_tail": prompt_texts[i][-500:],
        })
    err_records.sort(key=lambda r: -r["confidence"])  # most confident wrong first

    with (OUT_DIR / "error_analysis.json").open("w") as f:
        json.dump({"n_errors": int(err_mask.sum()), "errors": err_records}, f, indent=2)

    # Print top 8 confidently-wrong samples (4 each direction).
    fn_records = [r for r in err_records if r["true_label"] == 1][:4]
    fp_records = [r for r in err_records if r["true_label"] == 0][:4]

    def _dump(label, recs):
        print(f"\n──── {label} (most confidently wrong) ────")
        for r in recs:
            print(f"  id={r['id']:>3}  prob={r['prob_hallucinated']:.3f}  "
                  f"resp_len={r['response_len']}  "
                  f"response={r['response'][:100]!r}")

    _dump("False negatives (true=hallucinated, predicted=truthful)", fn_records)
    _dump("False positives (true=truthful, predicted=hallucinated)", fp_records)

    print(f"\nFull error dump: {OUT_DIR / 'error_analysis.json'}")


if __name__ == "__main__":
    main()
