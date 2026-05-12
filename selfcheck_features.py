"""SelfCheckGPT-via-sampling features.

For each (prompt, response) pair in dataset.csv and test.csv:
  1. Re-prompt Qwen with the same prompt and generate 5 alternative responses
     at temperature 0.7 (truncated to 60 new tokens).
  2. Compute inter-sample agreement metrics:
       - mean pairwise BLEU-1 (unigram precision)
       - mean pairwise Jaccard similarity on 3-gram sets
       - mean pairwise cosine similarity between last-token hidden states
         of each alternative (re-feeding each through Qwen).
  3. Plus: mean BLEU between each alternative and the ORIGINAL labelled
     response (this is the canonical SelfCheckGPT signal — disagreement
     of resampled answers with the answer-under-test correlates with
     hallucination).

Output: logs/audits/selfcheck_features.npz with keys "train" (N_train, 6) and
"test" (N_test, 6). Loaded by solution.py when USE_SELFCHECK=1.

Run from smiles/repo/:
  python selfcheck_features.py             # full dataset (~25 min on 5880 Ada)
  python selfcheck_features.py --quick     # 50-sample smoke test
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import MAX_LENGTH, get_model_and_tokenizer

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"
N_SAMPLES = 5
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 60
OUT_PATH = Path("logs/audits/selfcheck_features.npz")


def jaccard_ngrams(a: str, b: str, n: int = 3) -> float:
    def ngrams(s: str) -> set[tuple[str, ...]]:
        toks = s.lower().split()
        return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}
    A, B = ngrams(a), ngrams(b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(len(A | B), 1)


def bleu1_unigram(reference: str, candidate: str) -> float:
    ref_counts = Counter(reference.lower().split())
    cand_tokens = candidate.lower().split()
    if not cand_tokens:
        return 0.0
    cand_counts = Counter(cand_tokens)
    overlap = sum(min(cand_counts[t], ref_counts[t]) for t in cand_counts)
    precision = overlap / len(cand_tokens)
    # Brevity penalty.
    bp = 1.0 if len(cand_tokens) >= len(ref_counts.values()) else math.exp(1 - len(ref_counts.values()) / max(len(cand_tokens), 1))
    return precision * bp


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


def last_token_embedding(text: str, model, tokenizer, device) -> np.ndarray:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        out = model(**enc)
    h = out.hidden_states[-1].float().cpu().numpy()[0]
    # Last real-token position (no padding because batch=1, no padding applied).
    mask = enc["attention_mask"].cpu().numpy()[0]
    last_idx = int(mask.nonzero()[0][-1])
    return h[last_idx]


def features_for_row(prompt: str, original_response: str, model, tokenizer, device) -> np.ndarray:
    """Returns shape (6,)."""
    # 1. Generate N_SAMPLES alternative responses.
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            do_sample=True,
            temperature=TEMPERATURE,
            num_return_sequences=N_SAMPLES,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the generated suffix per sample.
    prompt_len = enc["input_ids"].shape[-1]
    alts = []
    for s in out_ids:
        gen = s[prompt_len:].tolist()
        alts.append(tokenizer.decode(gen, skip_special_tokens=True).strip())

    # 2. Pairwise inter-alt metrics.
    bleus = []
    jacc = []
    for i in range(N_SAMPLES):
        for j in range(i + 1, N_SAMPLES):
            bleus.append(bleu1_unigram(alts[i], alts[j]))
            jacc.append(jaccard_ngrams(alts[i], alts[j]))
    pair_bleu = float(np.mean(bleus)) if bleus else 0.0
    pair_jacc = float(np.mean(jacc)) if jacc else 0.0

    # 3. Pairwise embedding cosine.
    embs = np.stack([last_token_embedding(a, model, tokenizer, device) for a in alts])
    cos_sims = []
    for i in range(N_SAMPLES):
        for j in range(i + 1, N_SAMPLES):
            cos_sims.append(cosine(embs[i], embs[j]))
    pair_cos = float(np.mean(cos_sims)) if cos_sims else 0.0

    # 4. Each alt vs the labelled response (canonical SelfCheckGPT signal).
    orig_response_clean = original_response.replace("<|endoftext|>", "").strip()
    bleu_vs_orig = float(np.mean([bleu1_unigram(orig_response_clean, a) for a in alts]))
    jacc_vs_orig = float(np.mean([jaccard_ngrams(orig_response_clean, a) for a in alts]))

    # 5. Variance of pairwise cosines (consistency-of-consistency).
    cos_std = float(np.std(cos_sims)) if cos_sims else 0.0

    return np.array(
        [pair_bleu, pair_jacc, pair_cos, bleu_vs_orig, jacc_vs_orig, cos_std],
        dtype=np.float32,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="only process 50 samples for smoke test")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  N_SAMPLES={N_SAMPLES}  TEMP={TEMPERATURE}  MAX_NEW={MAX_NEW_TOKENS}")

    print("[Model] loading ...")
    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    df = pd.read_csv(DATA_FILE)
    df_test = pd.read_csv(TEST_FILE)
    if args.quick:
        df = df.head(50)
        df_test = df_test.head(20)

    print(f"[Train] {len(df)} samples")
    train_features = []
    t0 = time.time()
    for i, row in tqdm(df.iterrows(), total=len(df), desc="train SelfCheck"):
        try:
            f = features_for_row(row["prompt"], row["response"], model, tokenizer, device)
        except Exception as exc:  # generation failure on a single sample shouldn't kill the run
            print(f"  ! sample {i}: {exc!r}; falling back to zeros")
            f = np.zeros(6, dtype=np.float32)
        train_features.append(f)
    train_features = np.stack(train_features)
    print(f"  done in {time.time() - t0:.0f}s; train features shape {train_features.shape}")

    print(f"[Test] {len(df_test)} samples")
    test_features = []
    t1 = time.time()
    for i, row in tqdm(df_test.iterrows(), total=len(df_test), desc="test SelfCheck"):
        try:
            f = features_for_row(row["prompt"], row["response"], model, tokenizer, device)
        except Exception as exc:
            print(f"  ! sample {i}: {exc!r}; falling back to zeros")
            f = np.zeros(6, dtype=np.float32)
        test_features.append(f)
    test_features = np.stack(test_features)
    print(f"  done in {time.time() - t1:.0f}s; test features shape {test_features.shape}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, train=train_features, test=test_features)
    print(f"\nSaved to {OUT_PATH}")
    print("  feature columns: [pair_bleu, pair_jacc, pair_cos, bleu_vs_orig, jacc_vs_orig, cos_std]")


if __name__ == "__main__":
    main()
