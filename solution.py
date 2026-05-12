"""
Hallucination Detection in Small Language Models

# Files you can edit:
    - aggregation.py — layer selection and token pooling 
    - aggregation.py | extract_geometric_features — optional hand-crafted features 
    - probe.py | HallucinationProbe — probe classifier (nn.Module subclass) 
    - splitting.py | split_data — train / validation / test split strategy 

# Fixed infrastructure (do not edit)
    - model.py | LLM loader (get_model_and_tokenizer) 
    - evaluate.py | Evaluation loop, summary table, JSON output 

# Data Format — ChatML and Special Tokens
    The `prompt` column uses ChatML (Chat Markup Language), the conversation
    template built into Qwen models.  Each message is wrapped in role markers:

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    ... question and context ... <|im_end|>
    <|im_start|>assistant

    Special tokens and their roles:

    - `<|im_start|>` — opens a chat turn; the role (`system`, `user`, or `assistant`) immediately follows
    - `<|im_end|>` — closes the current chat turn
    - `<|endoftext|>` — end-of-sequence (EOS) token appended by the model at the end of its response

    The `prompt` ends right after `<|im_start|>assistant\n` — it provides the
    full context up to (but not including) the model's reply.  The `response`
    column holds the actual generated text, ending with `<|endoftext|>`.

    We feed the concatenation of `prompt + response` to the feature extractor
    so the hidden states capture both the question context and the model's
    specific answer — the hallucination signal lives in that joint representation.


"""

import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import wandb_utils
from aggregation import (
    AGG_LAYER,
    AGG_STRATEGY,
    aggregation_and_feature_extraction,
    compute_knn_features,
    compute_lid_features,
    compute_mahalanobis_features,
    extract_heuristic_features,
    extract_logit_lens_features,
    extract_per_head_attention_to_prompt,
    extract_perplexity_features,
)
from evaluate import print_summary, run_evaluation, save_predictions, save_results
from model import MAX_LENGTH, _DEFAULT_MODEL, get_model_and_tokenizer
from probe import PROBE_ARCH, PROBE_FAMILY, PROBE_WEIGHT_DECAY, HallucinationProbe
from splitting import SPLIT_SEED, SPLIT_STRATEGY, split_data

# ---------------------------------------------------------------------

DATA_FILE     = "./data/dataset.csv"   # path to the dataset CSV
OUTPUT_FILE   = "results.json"         # where to write the results summary
BATCH_SIZE    = 4
USE_GEOMETRIC = os.environ.get("USE_GEOMETRIC", "0") == "1"  # toggle via env var
USE_ATTENTION = os.environ.get("USE_ATTENTION", "0") == "1"  # if True, model outputs attention weights and we extract attention features
USE_PERPLEXITY = os.environ.get("USE_PERPLEXITY", "0") == "1"  # if True, extract per-token NLL stats over the response span
USE_HEURISTIC = os.environ.get("USE_HEURISTIC", "0") == "1"  # if True, append text-based heuristic features (length, hedge-words, etc.)
USE_LOGIT_LENS = os.environ.get("USE_LOGIT_LENS", "0") == "1"  # if True, project per-layer residuals through LM head and emit prediction-stability features
LOGIT_LENS_NORM = os.environ.get("LOGIT_LENS_NORM", "1") == "1"  # whether to apply the model's final RMSNorm before the LM head (the "with-norm" lens; more accurate)
USE_KNN_OOD = os.environ.get("USE_KNN_OOD", "0") == "1"  # if True, append per-sample KNN-OOD distance features (post-extraction)
USE_MANIFOLD = os.environ.get("USE_MANIFOLD", "0") == "1"  # if True, append LID (TwoNN) + per-class Mahalanobis distance features
USE_ATTENTION_TO_PROMPT = os.environ.get("USE_ATTENTION_TO_PROMPT", "0") == "1"  # if True (with USE_ATTENTION=1), add per-layer attention-mass-to-prompt features
USE_PER_HEAD_ATTN = os.environ.get("USE_PER_HEAD_ATTN", "0") == "1"  # if True (with USE_ATTENTION=1), add 336 per-(layer, head) attention-to-prompt features
USE_SELFCHECK = os.environ.get("USE_SELFCHECK", "0") == "1"  # if True, load pre-computed SelfCheckGPT features from logs/audits/selfcheck_features.npz
# Sweep knob. "prompt_response" (default, fed to Qwen as the original solution did),
# "prompt_only" (no model response — abstention-style probing), "response_only".
TEXT_MODE = os.environ.get("TEXT_MODE", "prompt_response")


def _build_text(row) -> str:
    if TEXT_MODE == "prompt_response":
        return f"{row['prompt']}{row['response']}"
    if TEXT_MODE == "prompt_only":
        return row["prompt"]
    if TEXT_MODE == "response_only":
        return row["response"]
    raise ValueError(f"unknown TEXT_MODE={TEXT_MODE!r}")
TEST_FILE        = "./data/test.csv"   # competition test set (labels are null)
PREDICTIONS_FILE = "predictions.csv"   # output file with predicted labels

assert OUTPUT_FILE == "results.json"
assert PREDICTIONS_FILE == "predictions.csv"
# ---------------------------------------------------------------------
if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device       : {device}")
    print(f"Data         : {DATA_FILE}")
    print(f"Max length   : {MAX_LENGTH} tokens")
    print(f"Geometric feats: {USE_GEOMETRIC}")


    df = pd.read_csv(DATA_FILE)

    # Build the text fed to the LLM (TEXT_MODE controls prompt+response, prompt-only, etc.).
    all_texts  = [_build_text(row) for _, row in df.iterrows()]
    all_labels = np.array([int(float(h)) for h in df["label"]])

    n_total = len(all_labels)
    n_hallucinated = int(all_labels.sum())
    n_truthful = int((all_labels == 0).sum())
    print(f"Loaded {n_total} samples  "
        f"({n_hallucinated} hallucinated / {n_truthful} truthful)")

    # ── wandb run init ────────────────────────────────────────────────────
    # Captures all knobs the student is likely to sweep over. Add fields here
    # if you introduce new hyperparameters in aggregation/probe/splitting.
    gpu_names = (
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    wandb_utils.init_run(
        config={
            "model_name": _DEFAULT_MODEL,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "use_geometric": USE_GEOMETRIC,
            "device": str(device),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": gpu_names,
            "data_file": DATA_FILE,
            "test_file": TEST_FILE,
            "n_samples": int(n_total),
            "n_hallucinated": n_hallucinated,
            "n_truthful": n_truthful,
            "class_balance_pos": float(n_hallucinated / max(n_total, 1)),
            "agg_strategy": AGG_STRATEGY,
            "agg_layer": AGG_LAYER,
            "probe_arch": PROBE_ARCH,
            "probe_weight_decay": PROBE_WEIGHT_DECAY,
            "probe_family": PROBE_FAMILY,
            "split_strategy": SPLIT_STRATEGY,
            "split_seed": SPLIT_SEED,
            "text_mode": TEXT_MODE,
        },
        tags=["geometric" if USE_GEOMETRIC else "no-geometric"],
    )
    
    # Preview the raw data
    print(f"Columns : {df.columns.tolist()}")
    print(f"Rows    : {len(df)}")
    print(f"Labels  : {dict(df['label'].value_counts().sort_index())}")
    print()

    # Show the first sample (truncated for readability)
    row0 = df.iloc[0]
    print("── prompt (first 500 chars) " + "─" * 34)
    print(row0["prompt"][:500])
    print()
    print("── response (first 300 chars) " + "─" * 31)
    print(row0["response"][:300])
    print()
    label_str = "hallucinated" if int(row0["label"]) else "truthful"
    print(f"── label : {int(row0['label'])}  ({label_str})")


    # Load the LLM. If attention features are requested, reload with eager
    # attention since SDPA doesn't support output_attentions=True.
    model, tokenizer = get_model_and_tokenizer()
    if USE_ATTENTION:
        from transformers import AutoModelForCausalLM as _AMLM
        print("[Model] reloading with attn_implementation='eager' for USE_ATTENTION")
        model = _AMLM.from_pretrained(
            _DEFAULT_MODEL,
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    # Precompute per-sample prompt token counts (needed by perplexity, logit-lens,
    # attention-to-prompt, AND per-head probing to identify the response span).
    if USE_PERPLEXITY or USE_LOGIT_LENS or USE_ATTENTION_TO_PROMPT or USE_PER_HEAD_ATTN:
        prompt_token_counts = [
            len(tokenizer.encode(r["prompt"], add_special_tokens=False))
            for _, r in df.iterrows()
        ]
    else:
        prompt_token_counts = [0] * len(all_texts)

    # Cache LM-head weight (cast to fp32 ONCE; ~540MB tensor). NOTE: held on
    # CPU because this host's NVML/driver mismatch causes the CUDA caching
    # allocator to assert on log_softmax over the full 151K vocab. CPU is
    # slower (~1s per sample) but reliable.
    if USE_LOGIT_LENS:
        lm_head_weight = model.lm_head.weight.detach().float().cpu().contiguous()
        # Mirror the final-norm module to CPU so the lens applies it consistently.
        import copy
        final_norm_module = copy.deepcopy(model.model.norm).cpu() if LOGIT_LENS_NORM else None
    else:
        lm_head_weight = None
        final_norm_module = None

    all_features: list = []
    t0 = time.time()

    for start in tqdm(range(0, len(all_texts), BATCH_SIZE),
                    desc="Extracting & aggregating", unit="batch"):

        # ── 1. Tokenise the current mini-batch ───────────────────────────────
        batch_texts = all_texts[start : start + BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=USE_ATTENTION,
            )

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        # (batch, n_layers, n_heads, seq_len, seq_len) if attentions are requested.
        attn = torch.stack(outputs.attentions, dim=1).float() if USE_ATTENTION else None
        logits = outputs.logits.float() if USE_PERPLEXITY else None
        mask = attention_mask.cpu()

        for i in range(hidden.size(0)):
            feat = aggregation_and_feature_extraction(
                hidden[i],
                mask[i],
                attentions=attn[i] if attn is not None else None,
                use_geometric=USE_GEOMETRIC,
                prompt_token_count=prompt_token_counts[start + i] if USE_ATTENTION_TO_PROMPT else None,
            ).cpu()
            if USE_PERPLEXITY:
                ppl_feat = extract_perplexity_features(
                    logits[i],
                    input_ids[i],
                    attention_mask[i].cpu(),
                    prompt_token_counts[start + i],
                ).cpu()
                feat = torch.cat([feat, ppl_feat], dim=0)
            if USE_LOGIT_LENS:
                ll_feat = extract_logit_lens_features(
                    hidden[i].cpu(),                            # move to CPU for the lens projection
                    input_ids[i].cpu(),
                    attention_mask[i].cpu(),
                    prompt_token_counts[start + i],
                    lm_head_weight,                              # already on CPU
                    final_norm_module,
                )
                feat = torch.cat([feat, ll_feat], dim=0)
            if USE_PER_HEAD_ATTN and attn is not None:
                ph_feat = extract_per_head_attention_to_prompt(
                    attn[i],
                    attention_mask[i].cpu(),
                    prompt_token_counts[start + i],
                ).cpu()
                feat = torch.cat([feat, ph_feat], dim=0)
            if USE_HEURISTIC:
                row = df.iloc[start + i]
                heur_feat = extract_heuristic_features(row["prompt"], row["response"])
                feat = torch.cat([feat, heur_feat], dim=0)
            all_features.append(feat)

    extract_time = time.time() - t0
    print(f"Done in {extract_time:.1f} s  —  {len(all_features)} feature vectors extracted")

    # Stack into the (N, feature_dim) matrix used by the probe.
    X = np.vstack([f.numpy() for f in all_features])   # shape: (N, feature_dim)
    y = all_labels                                       # shape: (N,)

    # Keep a pre-augmentation copy so X_test's manifold features can be
    # computed against the same neighbor pool that training saw.
    X_base_for_knn = X.copy() if (USE_KNN_OOD or USE_MANIFOLD) else None

    if USE_KNN_OOD:
        knn_train = compute_knn_features(X, leave_one_out=True)
        X = np.hstack([X, knn_train])
        print(f"KNN-OOD: appended {knn_train.shape[1]} features (train, LOO)")

    if USE_MANIFOLD:
        # NOTE: Mahalanobis dropped — it has label leakage when X_query is in X_train
        # (each sample's label was used to fit its own class's Gaussian). LID is
        # label-free and stays. See SOLUTION.md "leakage and lessons" section.
        lid_train = compute_lid_features(X_base_for_knn, leave_one_out=True)
        X = np.hstack([X, lid_train])
        print(f"Manifold: appended LID ({lid_train.shape[1]}) features (train) — Mahalanobis dropped due to label leakage")

    if USE_SELFCHECK:
        sc_path = "logs/audits/selfcheck_features.npz"
        sc = np.load(sc_path)
        sc_train = sc["train"]
        assert sc_train.shape[0] == X.shape[0], (
            f"SelfCheck train features have {sc_train.shape[0]} rows; expected {X.shape[0]}. "
            f"Re-run selfcheck_features.py without --quick."
        )
        X = np.hstack([X, sc_train])
        print(f"SelfCheck: appended {sc_train.shape[1]} features (train)")

    print(f"Feature matrix : {X.shape}  (feature_dim = {X.shape[1]})")
    print(f"Geometric feats: {USE_GEOMETRIC}")

    splits = split_data(y, df)

    print(f"Splits : {len(splits)} fold(s)")
    for i, (tr, va, te) in enumerate(splits):
        print(f"  Fold {i + 1}: train={len(tr)}  "
            f"val={len(va) if va is not None else 'N/A'}  test={len(te)}")

    wandb_utils.log(
        {
            "extract/time_s": extract_time,
            "extract/n_features": int(X.shape[0]),
            "extract/feature_dim": int(X.shape[1]),
            "extract/sec_per_sample": extract_time / max(len(all_features), 1),
            "splits/n_folds": len(splits),
        }
    )

    fold_results = run_evaluation(splits, X, y, HallucinationProbe)

    # ── Per-fold metrics to wandb ────────────────────────────────────────
    for r in fold_results:
        fold = r["fold"]
        scalar_metrics = {
            f"fold_{fold}/{k}": v
            for k, v in r.items()
            if isinstance(v, (int, float)) and k != "fold"
        }
        wandb_utils.log(scalar_metrics)
    wandb_utils.log_fold_table(fold_results)

    print_summary(fold_results, X.shape[1], len(X), extract_time)
    save_results(fold_results, X.shape[1], len(X), extract_time, OUTPUT_FILE)

    # ── Averaged / summary metrics (these show up in the wandb runs table) ─
    def _mean(key: str) -> float:
        vals = [r[key] for r in fold_results if key in r and r[key] == r[key]]  # filter NaN
        return float(np.mean(vals)) if vals else float("nan")

    wandb_utils.update_summary(
        {
            "avg/baseline_accuracy": _mean("baseline_accuracy"),
            "avg/baseline_f1": _mean("baseline_f1"),
            "avg/train_accuracy": _mean("train_accuracy"),
            "avg/train_f1": _mean("train_f1"),
            "avg/train_auroc": _mean("train_auroc"),
            "avg/val_accuracy": _mean("val_accuracy"),
            "avg/val_f1": _mean("val_f1"),
            "avg/val_auroc": _mean("val_auroc"),
            "avg/test_accuracy": _mean("test_accuracy"),
            "avg/test_f1": _mean("test_f1"),
            "avg/test_auroc": _mean("test_auroc"),
            "feature_dim": int(X.shape[1]),
            "extract_time_s": extract_time,
        }
    )

    

    # ── Load test data ────────────────────────────────────────────────────────
    df_test    = pd.read_csv(TEST_FILE)
    test_texts = [_build_text(row) for _, row in df_test.iterrows()]
    test_ids   = df_test.index
    print(f"Test set loaded: {len(test_texts)} samples")
    wandb_utils.log({"test/n_samples": len(test_texts)})
    t_test0 = time.time()

    # ── Extract features for test set (same loop as Section 4) ───────────────
    test_features: list = []

    if USE_PERPLEXITY or USE_LOGIT_LENS or USE_ATTENTION_TO_PROMPT or USE_PER_HEAD_ATTN:
        test_prompt_token_counts = [
            len(tokenizer.encode(r["prompt"], add_special_tokens=False))
            for _, r in df_test.iterrows()
        ]
    else:
        test_prompt_token_counts = [0] * len(test_texts)

    for start in tqdm(range(0, len(test_texts), BATCH_SIZE),
                    desc="Test extraction & aggregation", unit="batch"):

        batch_texts = test_texts[start : start + BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=USE_ATTENTION,
            )

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        attn = torch.stack(outputs.attentions, dim=1).float() if USE_ATTENTION else None
        logits = outputs.logits.float() if USE_PERPLEXITY else None
        mask = attention_mask.cpu()

        for i in range(hidden.size(0)):
            feat = aggregation_and_feature_extraction(
                hidden[i],
                mask[i],
                attentions=attn[i] if attn is not None else None,
                use_geometric=USE_GEOMETRIC,
                prompt_token_count=test_prompt_token_counts[start + i] if USE_ATTENTION_TO_PROMPT else None,
            ).cpu()
            if USE_PERPLEXITY:
                ppl_feat = extract_perplexity_features(
                    logits[i],
                    input_ids[i],
                    attention_mask[i].cpu(),
                    test_prompt_token_counts[start + i],
                ).cpu()
                feat = torch.cat([feat, ppl_feat], dim=0)
            if USE_LOGIT_LENS:
                ll_feat = extract_logit_lens_features(
                    hidden[i].cpu(),
                    input_ids[i].cpu(),
                    attention_mask[i].cpu(),
                    test_prompt_token_counts[start + i],
                    lm_head_weight,
                    final_norm_module,
                )
                feat = torch.cat([feat, ll_feat], dim=0)
            if USE_PER_HEAD_ATTN and attn is not None:
                ph_feat = extract_per_head_attention_to_prompt(
                    attn[i],
                    attention_mask[i].cpu(),
                    test_prompt_token_counts[start + i],
                ).cpu()
                feat = torch.cat([feat, ph_feat], dim=0)
            if USE_HEURISTIC:
                row = df_test.iloc[start + i]
                heur_feat = extract_heuristic_features(row["prompt"], row["response"])
                feat = torch.cat([feat, heur_feat], dim=0)
            test_features.append(feat)

    X_test = np.vstack([f.numpy() for f in test_features])  # (n_test, feature_dim)
    # Keep pre-augmentation X_test as the query input for KNN/manifold lookups
    # (these query against the pre-augmentation X_base_for_knn).
    X_test_base = X_test.copy() if (USE_KNN_OOD or USE_MANIFOLD) else None

    if USE_KNN_OOD:
        knn_test = compute_knn_features(X_base_for_knn, X_query=X_test_base, leave_one_out=False)
        X_test = np.hstack([X_test, knn_test])
        print(f"KNN-OOD: appended {knn_test.shape[1]} features (test, against training pool)")
    if USE_MANIFOLD:
        lid_test = compute_lid_features(X_base_for_knn, X_query=X_test_base, leave_one_out=False)
        X_test = np.hstack([X_test, lid_test])
        print(f"Manifold: appended LID ({lid_test.shape[1]}) features (test)")
    if USE_SELFCHECK:
        sc_path = "logs/audits/selfcheck_features.npz"
        sc = np.load(sc_path)
        sc_test = sc["test"]
        assert sc_test.shape[0] == X_test.shape[0], (
            f"SelfCheck test features have {sc_test.shape[0]} rows; expected {X_test.shape[0]}. "
            f"Re-run selfcheck_features.py without --quick."
        )
        X_test = np.hstack([X_test, sc_test])
        print(f"SelfCheck: appended {sc_test.shape[1]} features (test)")
    test_extract_time = time.time() - t_test0
    wandb_utils.log({"test/extract_time_s": test_extract_time})

    # ── Fit final probe on training + validation data only ──────────────────
    # Collect the union of all train and validation indices across every split.
    # For a single split this excludes idx_test; for k-fold every sample appears
    # in a training fold, so all samples are used (same as fitting on X, y).
    idx_non_test = np.unique(np.concatenate([
        np.concatenate([idx_tr, idx_va]) if idx_va is not None else idx_tr
        for idx_tr, idx_va, _ in splits
    ]))

    # Hold out 10% of idx_non_test as a final-stage val set for threshold
    # tuning of the submission probe. Without this, the final probe would use
    # threshold=0.5 and ignore the accuracy-mode tuner we use in evaluation.
    from sklearn.model_selection import train_test_split as _final_tts
    idx_ft, idx_fv = _final_tts(
        idx_non_test, test_size=0.10, random_state=SPLIT_SEED,
        stratify=y[idx_non_test],
    )
    final_probe = HallucinationProbe()
    final_probe.fit(X[idx_ft], y[idx_ft])
    final_probe.fit_hyperparameters(X[idx_fv], y[idx_fv])
    print(f"Final probe: trained on {len(idx_ft)} samples, "
          f"threshold tuned on {len(idx_fv)} (chosen threshold: {final_probe._threshold:.4f})")
    wandb_utils.log({
        "final_probe/n_train": len(idx_ft),
        "final_probe/n_val": len(idx_fv),
        "final_probe/threshold": float(final_probe._threshold),
    })

    # ── Predict and save ────────────────────────────────────────────────────
    save_predictions(final_probe, X_test, test_ids, PREDICTIONS_FILE)

    # Log prediction distribution + persist artifacts to wandb (predictions
    # are how the competition is scored, so keep a copy attached to the run).
    try:
        preds = pd.read_csv(PREDICTIONS_FILE)
        wandb_utils.update_summary(
            {
                "predictions/n": int(len(preds)),
                "predictions/frac_hallucinated": float(preds["label"].mean()),
            }
        )
        if wandb_utils.is_active():
            import wandb  # local import keeps the file importable without wandb
            artifact = wandb.Artifact("submission", type="predictions")
            artifact.add_file(PREDICTIONS_FILE)
            artifact.add_file(OUTPUT_FILE)
            wandb.run.log_artifact(artifact)
    except Exception as exc:  # pragma: no cover -- artifact upload is best-effort
        print(f"[wandb] failed to upload artifact: {exc}")

    wandb_utils.finish_run()

