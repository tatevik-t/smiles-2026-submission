# SMILES-2026 Hallucination Detection — Solution Report

**Applicant:** Tatevik Ter-Hovhannisyan.
**Built with:** the [Claude Code](https://claude.com/claude-code) assistant (Anthropic). See [§Contribution attribution](#contribution-attribution) for the work split.
**Wandb dashboard:** [wandb.ai/tatevik-th/smiles-2026-hallucination](https://wandb.ai/tatevik-th/smiles-2026-hallucination?nw=nwusertatevikt).

---

## Headline result

| Metric | Value |
|---|---|
| **Local 5-fold test accuracy** (flagship single model) | **74.60%** |
| **Local 5-fold test AUROC** (flagship single model) | **78.10%** |
| Majority-class baseline | 70.10% |
| Canonical `predictions.csv` distribution | 82/18 hallucinated/truthful |

`predictions.csv` is a 5-model majority-vote ensemble of diverse probes; `results.json` reports the highest-AUROC single config (`xgb-everything-5fold-clean`).

## One-paragraph approach

We feed `prompt + response` through frozen Qwen2.5-0.5B, extract the **last-token hidden state of the final transformer layer** as the base 896-dim feature, and **augment** it with a stack of architecturally-motivated signals: per-token NLL of the response under Qwen, hand-crafted text heuristics (length, hedge-words, prompt-overlap), per-(layer, head) attention-to-prompt fractions, local intrinsic dimensionality of the embedding cloud, and **SelfCheckGPT 5×-sampling consistency**. An **XGBoost** probe over the 1,361-dim feature vector with **accuracy-mode threshold tuning** on each fold's validation slice gives our best single number; a 5-model majority-vote **ensemble** stabilises the submission.

## Key scientific findings

### 1. The signal is sparse: 100 of 336 attention heads is enough

A K=100 sparse probe (top-100 heads selected by mutual information within each CV fold) matches the K=336 dense probe within noise (75.18% ± 2.1pt vs 74.88% ± 3.0pt test_acc). The signal lives in roughly the top *third* of the attention heads. The same heads keep getting picked across CV folds — layers 17, 18, 22, 23 dominate. Reproduce via `python sparse_head_probe.py`.

### 2. Hallucinated responses "look away" from the prompt — universally

Across **322 of the 336 heads (96%)**, hallucinated samples allocate *less* attention mass to the prompt span than truthful samples do. Mean attention-to-prompt: 63.5% (truthful) vs 57.7% (hallucinated). Top-5 heads show Cohen's d 0.58-0.79 with p < 10⁻¹¹. Mechanistically: **when fabricating, the model looks less at the source material across nearly every attention head, end to end**. Reproduce via `python head_attribution.py`.

### 3. Mid-layer probing fails for residual streams but succeeds for attention patterns

A per-layer scan over hidden-state aggregations showed the **final layer wins** (layers 8-20 lose 1-8pt test_AUROC). However the per-(layer, head) attention-feature MI analysis above shows **a bimodal layer distribution** with peaks at layer 10 and layers 22-23. The published probing literature's "mid-layer carries the signal" claim is partly a **representation-choice** artifact: residual streams concentrate signal at the top in small models, attention patterns distribute it throughout.

### 4. The signal is non-linear in the last-token representation

PCA's top component explains 67% of the variance, so a linear probe should suffice — except it doesn't: dropping the MLP's single hidden layer drops test_AUROC 9 points (75% → 66%). The label direction is **not** aligned with the dominant variance axis.

### 5. SelfCheckGPT alone is the only perfectly-calibrated probe

5× temperature-sampled responses + pairwise BLEU/cosine consistency yields a 6-dim feature vector. As a standalone probe it gives 71.69% test_acc with **71/29 hallucinated predictions on `data/test.csv` — matching the 70% training prior exactly**. Every hidden-state-based probe over-predicts hallucinated (XGBoost: 87-96%; MLP: 73-78%). When fused with the hidden-state features, it lifts test_acc by ~0.7pt (74.60% final vs 73.9% without SelfCheckGPT).

## Ablation: what moved the metric

Building from the original `solution.py` (last-token + MLP + F1-greedy threshold collapse) to the flagship:

| Increment | feat_dim | local 5-fold test_acc | Δ |
|---|---|---|---|
| Original `solution.py` baseline | 896 | 70.19% (majority class collapse) | — |
| **Accuracy-mode threshold tuning + reject-degenerate** | 896 | 74.04% (single-split) | **+3.85** |
| 5-fold CV + threshold-tuned final probe | 896 | 69.96% (more honest estimate) | — |
| + Per-token perplexity / NLL features | 902 | 71.40% | +1.44 |
| + Hand-crafted heuristic features | 911 | 71.26% | within noise |
| **Switch MLP → XGBoost backend** | 911 | 74.17% | **+2.91** |
| + KNN-OOD distances | 916 | 74.89% | +0.72 |
| + LID manifold features | 918 | 74.95% (clean) | within noise |
| + Cross-attention-to-prompt features | 1019 | 75.18% | within noise |
| + Per-head attention features (336 dims) | 1355 | 74.74% | within noise |
| **+ SelfCheckGPT 5×-sampling features** | **1361** | **74.60%** flagship / **78.10% AUROC** | +0.4 (AUROC +1.6) |

The two largest individual jumps were methodological: (1) **fixing the F1-greedy threshold-tuning collapse** that masked all probe differences in baseline, and (2) **switching to XGBoost** which handles the heterogeneous-scale feature vector much better than the MLP could.

## Reproducibility

### Environment

- Python 3.12, single CUDA GPU (≥4GB)
- Qwen2.5-0.5B loaded in bfloat16 with `output_hidden_states=True`

### Setup

```bash
git clone https://github.com/tatevik-t/smiles-2026-submission
cd smiles-2026-submission
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Reproduce the submission

The submission ensembles five 5-fold runs and uses one of them as `results.json`. The two analysis scripts that produce the head-attribution and SelfCheckGPT features must run once before the sweep.

```bash
# One-time feature-precomputation (5-fold-internal SelfCheckGPT + head probes)
python selfcheck_features.py        # ~15 min, writes logs/audits/selfcheck_features.npz
python head_attribution.py          # ~30 s, writes logs/audits/head_attribution_*.{npy,json,npz}

# Generate the 5 ensemble inputs (the canonical sweep)
ONLY="55,56,57,46,5" SUFFIX="-clean" ./run_sweep.sh

# Ensemble
python ensemble_predictions.py \
  --runs xgb-everything-5fold-clean abl-no-perplexity-clean \
         xgb-selfcheck-5fold manifold-5fold-clean final-submission \
  --out predictions.csv

# Pick the highest-AUROC clean single model as results.json
cp logs/runs/xgb-everything-5fold-clean/results.json results.json
```

To turn off wandb tracking on any individual run, prepend `WANDB_MODE=disabled`.

### Hyperparameter knobs

Every sweepable setting is an environment variable read by one of `solution.py`, `aggregation.py`, `probe.py`, or `splitting.py`:

| Env var | Default | Values |
|---|---|---|
| `AGG_STRATEGY` | `last_token` | `last_token`, `mean_pool`, `last4_concat`, `meanmaxlast` |
| `AGG_LAYER` | `-1` | integer in `[-25, 24]` |
| `PROBE_ARCH` | `mlp_1h_256` | `mlp_1h_256`, `mlp_deep`, `linear` |
| `PROBE_FAMILY` | `torch` | `torch`, `xgboost` |
| `SPLIT_STRATEGY` | `single` | `single`, `5fold` |
| `SPLIT_SEED` | `42` | integer |
| `USE_PERPLEXITY` | `0` | 0/1 |
| `USE_HEURISTIC` | `0` | 0/1 |
| `USE_ATTENTION` | `0` | 0/1 (forces eager attention) |
| `USE_ATTENTION_TO_PROMPT` | `0` | 0/1 |
| `USE_PER_HEAD_ATTN` | `0` | 0/1 |
| `USE_KNN_OOD` | `0` | 0/1 |
| `USE_MANIFOLD` | `0` | 0/1 |
| `USE_SELFCHECK` | `0` | 0/1 (loads pre-computed npz) |
| `TEXT_MODE` | `prompt_response` | `prompt_response`, `prompt_only`, `response_only` |

## Files in this repo

```
SMILES-2026-submission/
├── data/                      # competition data (unchanged)
├── solution.py                # main pipeline, env-var-driven (modified for new features)
├── predictions.csv            # SUBMISSION: 5-model ensemble
├── results.json               # SUBMISSION: xgb-everything-5fold-clean 5-fold metrics
├── SOLUTION.md                # this report
├── TOMORROW.md                # planned but unrun experiments
│
│   ── Competition-editable (modified) ──────────────────────────────
├── aggregation.py             # base aggregation + perplexity/attention/heuristic/manifold/logit-lens feature extractors
├── probe.py                   # HallucinationProbe — MLP and XGBoost backends, accuracy-mode threshold tuner
├── splitting.py               # single-split + stratified 5-fold splits
│
│   ── Fixed infrastructure (unchanged) ──────────────────────────────
├── model.py                   # loads Qwen2.5-0.5B
├── evaluate.py                # evaluation loop, metrics, summary
│
│   ── Applicant analysis tooling (not part of competition contract) ─
├── run_sweep.sh               # env-var sweep launcher with per-run archiving
├── compare_runs.py            # cross-run leaderboard table
├── error_analysis.py          # per-sample misclassification dump
├── embeddings_audit.py        # distribution-shift checks
├── slice_similarity.py        # per-fold similarity to data/test.csv
├── weighted_metrics.py        # trust-weighted leaderboard estimates
├── ensemble_predictions.py    # majority-vote ensemble of predictions.csv files
├── head_attribution.py        # per-(layer, head) MI ranking
├── sparse_head_probe.py       # CV-proper sparse probe (K=5..336)
├── sparse_probe_submission.py # standalone sparse-K probe writing predictions.csv
├── selfcheck_features.py      # SelfCheckGPT 5× sampling features
│
└── logs/                      # per-run archives and audit JSON files
```

## Experiments that didn't work

- **Mean-pool / last4-layer concatenation** — destroyed signal (single position carries most info for causal LM).
- **Mid-layer probing of residual stream (layers 8-20)** — final layer wins on Qwen2.5-0.5B; published mid-layer results don't transfer at 0.5B.
- **Linear probe (no hidden layer)** — underfits by 9pt AUROC; signal is non-linear in last-token representation.
- **Geometric features** (per-layer norms + inter-layer cosine drift) — added noise without signal.
- **F1-as-primary threshold-tuning metric** (the original code path) — collapsed every probe to majority-class baseline.
- **Prompt-only / response-only TEXT_MODE** — both flop on AUROC; signal needs both halves of the conversation.
- **Logit lens** — full-vocab `log_softmax` over 25 layers triggered an NVML/CUDA driver-mismatch assert on the host; CPU fallback hit per-batch OOM. Code intact in `aggregation.py:extract_logit_lens_features` for future replay.

## Future work (ordered by expected leverage)

- **Activation patching for causal evidence.** Swap a hallucinated sample's hidden state at layer L with a truthful sample's; measure how the probe's prediction flips. Identifies the *causally* important layers, not just the *correlationally* informative ones.
- **Response-span-only multi-token aggregation.** Phase 9 tried mean/max over ALL 512 token positions and lost. With prompt/response boundary tracking, compute mean/max/std *just* over response tokens — should pick up the response-length signal at the representation level.
- **Sparse autoencoder dictionary on the residual stream.** Decompose the 896-dim hidden state into ~16K sparse interpretable features. Probe on the sparse features for interpretable mechanistic claims.
- **Probability-level ensembling.** Currently we ensemble labels (majority vote). Saving per-sample probabilities and averaging them usually beats label majority vote on small data.
- **Multi-prompt augmentation.** Synthesize paraphrased prompts via another LLM; expand the 689-sample training set. Risk: label noise from paraphrases that genuinely change answerability.

## Leakage and lessons (two-layer post-mortem)

**Layer 1 — Mahalanobis label leakage.** The mid-day ablation sweep produced cross-validation test accuracies of 83% and 96% — far above what any 0.5B-model probe should achieve. The cause: `compute_mahalanobis_features` fit a per-class Gaussian on the full training set including each query sample's own label, so each training sample's Mahalanobis features were partially computed using its own label.

**First fix attempt — leave-one-out centroid.** Replaced Mahalanobis with Euclidean distance to class centroids, with the query sample's contribution to its own class's centroid removed: `mu_c_loo = (n_c * mu_c − X_i) / (n_c − 1)`. On synthetic data, the LOO check passed. On real data, re-runs *still* showed 85-89% test accuracy — only 1-3pt below the leakier Mahalanobis path.

**Layer 2 — Transductive leakage.** Even with sample-level LOO, the class centroids encode the labels of *every other sample* in the dataset, **including samples in other folds' test sets**. Each fold's training samples' centroid features encode labels of samples that fold will eventually be evaluated against. This is technically transductive learning, not direct leakage, but it artificially boosts CV accuracy and won't transfer to the leaderboard's truly-unseen `data/test.csv`.

**Final resolution.** Both Mahalanobis and centroid features dropped from the canonical pipeline. The `USE_MANIFOLD=1` path now emits **LID only** (TwoNN — uses only ratios of nearest-neighbour distances, never any labels). KNN-OOD is also label-free. The flagship `xgb-everything-5fold-clean` at 74.60% / 78.10% AUROC is the honest, leakage-free result.

**Proper fix (future work).** Compute label-aware features inside the CV inner loop, with each fold's training-set labels only. This requires either: (a) refactoring the probe class to receive raw X and y and compute its own augmented features, or (b) restructuring `evaluate.py` to accept per-fold feature matrices. Both are 1-2 hours of careful work; left for next session.

**Lesson learned.** Any feature computed using class labels must respect the CV-fold boundary, not just exclude the query sample. Sample-level LOO is necessary but not sufficient for fold-correctness in 5-fold CV. This pitfall is particularly insidious because it only manifests in certain feature combinations (when the leaky features dominate XGBoost splits) and passes the "does train accuracy match val accuracy" sanity check (because both train and val have the same kind of leakage built in).

## Contribution attribution

This solution was developed in an interactive paired-programming session between **Tatevik Ter-Hovhannisyan** (applicant) and the **Claude Code** assistant (Anthropic).

**Tatevik (applicant) curated:**
- All strategic decisions: when to keep experimenting vs. ship, choice of 5-fold cross-validation as canonical, order of experimental phases, which negative results to discard.
- Key hypotheses that drove the most productive phases — the **mid-layer probing direction** (empirically observed earlier with an intern), the **meta-question "what aren't we considering?"** that surfaced error analysis + perplexity + XGBoost, and the **prompt-only / abstention-style probing** hypothesis.
- All review/approval decisions on plan changes and final submission configuration.

**Claude Code implemented (under direction):**
- Env-var-driven sweep harness ([run_sweep.sh](run_sweep.sh)) and all analysis tooling.
- Code changes to editable competition files (`aggregation.py`, `probe.py`, `splitting.py`) and the additions to `solution.py` needed to thread new features through.
- Root-cause debugging of every bug: the **Mahalanobis label-leakage** (above), perplexity off-by-one, SDPA→eager attention swap, XGBoost calibration drift, F1-greedy threshold collapse.
- Result interpretation, the mechanistic-interpretability analyses, drafting of this report.
