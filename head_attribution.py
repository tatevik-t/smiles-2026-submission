"""Identify which (layer, head) pairs in Qwen2.5-0.5B carry hallucination signal.

For each of the 24 transformer layers x 14 attention heads = 336 heads, compute:
  1. Per-sample attention-mass-to-prompt-span at the last-response-token.
  2. Mutual information (MI) between this 1-D feature and the binary label.
  3. Rank heads by MI and report the top-K.

Outputs:
  - logs/audits/head_attribution.json: per-head MI table.
  - logs/audits/head_attribution_heatmap.npy: (24, 14) MI matrix for visualization.
  - logs/audits/head_attribution_topK.json: top-30 heads with their (layer, head, MI).

This is the mechanistic-interpretability analog of "find the induction heads" —
identifies the specific heads where the hallucination signal lives.

Run from smiles/repo/:
  python head_attribution.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from model import MAX_LENGTH, _DEFAULT_MODEL, get_model_and_tokenizer

DATA_FILE = "./data/dataset.csv"
BATCH_SIZE = 4
OUT_DIR = Path("logs/audits")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(DATA_FILE)
    y = np.array([int(float(h)) for h in df["label"]])
    n = len(y)
    print(f"Loaded {n} samples")

    # Load model with eager attention so we get per-head matrices.
    print(f"[Model] loading '{_DEFAULT_MODEL}' with attn_implementation='eager' ...")
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

    # Per-sample prompt token counts.
    prompt_token_counts = [
        len(tok.encode(r["prompt"], add_special_tokens=False))
        for _, r in df.iterrows()
    ]

    # Probe one forward to discover n_layers, n_heads.
    enc = tok([df.iloc[0]["prompt"] + df.iloc[0]["response"]], return_tensors="pt",
              truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        out = model(**enc)
    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]
    print(f"  n_layers={n_layers}  n_heads={n_heads}  -> {n_layers * n_heads} heads total")

    # Per-sample, per-head: attention-to-prompt mass averaged over response query positions.
    feats = np.zeros((n, n_layers, n_heads), dtype=np.float32)

    t0 = time.time()
    for start in tqdm(range(0, n, BATCH_SIZE), desc="head-attribution forward passes", unit="batch"):
        end = min(start + BATCH_SIZE, n)
        batch_texts = [df.iloc[i]["prompt"] + df.iloc[i]["response"] for i in range(start, end)]
        enc = tok(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc)
        # (n_layers, batch, n_heads, seq, seq) -> stack on layer dim
        attn = torch.stack(out.attentions, dim=0).float()                  # (n_layers, batch, n_heads, seq, seq)

        attention_mask = enc["attention_mask"].cpu()
        for bi in range(end - start):
            global_idx = start + bi
            mask = attention_mask[bi]
            last_pos = int(mask.nonzero(as_tuple=False)[-1].item())
            real_length = last_pos + 1
            ptc = prompt_token_counts[global_idx]
            if ptc <= 0 or ptc >= real_length:
                continue
            # (n_layers, n_heads, n_resp, real_length)
            slice_attn = attn[:, bi, :, ptc:real_length, :real_length]
            slice_attn = slice_attn / slice_attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            prompt_mass = slice_attn[..., :ptc].sum(dim=-1)                # (n_layers, n_heads, n_resp)
            feats[global_idx] = prompt_mass.mean(dim=-1).cpu().numpy()

    extract_time = time.time() - t0
    print(f"\nFeature extraction done in {extract_time:.1f}s.  feats shape: {feats.shape}")

    # Compute MI per (layer, head) feature vs label.
    flat = feats.reshape(n, n_layers * n_heads)
    print("Computing mutual information per head ...")
    mi = mutual_info_classif(flat, y, random_state=42, n_neighbors=5)
    mi_matrix = mi.reshape(n_layers, n_heads)

    # Top-K heads.
    flat_with_idx = [
        {"layer": int(l), "head": int(h), "mi": float(mi_matrix[l, h])}
        for l in range(n_layers) for h in range(n_heads)
    ]
    flat_with_idx.sort(key=lambda r: -r["mi"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "head_attribution_heatmap.npy", mi_matrix)
    # Save the raw per-sample per-head features so downstream sparse probing
    # can use them without re-running the model.
    np.savez(
        OUT_DIR / "head_attribution_features.npz",
        feats=feats.reshape(n, n_layers * n_heads),
        y=y,
        n_layers=np.array(n_layers),
        n_heads=np.array(n_heads),
    )
    with (OUT_DIR / "head_attribution_topK.json").open("w") as f:
        json.dump({"top_30": flat_with_idx[:30], "n_layers": n_layers, "n_heads": n_heads}, f, indent=2)
    print(f"Saved heatmap to {OUT_DIR / 'head_attribution_heatmap.npy'}")
    print(f"Saved top-30 to {OUT_DIR / 'head_attribution_topK.json'}")

    # Print top-15.
    print("\n──── Top 15 hallucination heads (by mutual information with label) ────")
    print(f"  {'rank':<5} {'layer':<6} {'head':<5} {'MI':>10}")
    print("  " + "-" * 33)
    for i, r in enumerate(flat_with_idx[:15]):
        print(f"  {i + 1:<5} {r['layer']:<6} {r['head']:<5} {r['mi']:>10.4f}")
    print(f"\n  Median MI across all 336 heads: {float(np.median(mi)):.4f}")
    print(f"  Max MI: {float(np.max(mi)):.4f}  Min MI: {float(np.min(mi)):.4f}")

    # Per-layer mean MI (which layers carry most signal?)
    print("\n  Per-layer mean MI across heads:")
    for l in range(n_layers):
        bar = "█" * int(mi_matrix[l].mean() * 200)
        print(f"    layer {l:>2}: {mi_matrix[l].mean():.4f}  {bar}")


if __name__ == "__main__":
    main()
