# 🔍 SMILES-2026 Hallucination Detection — Applicant Submission

Detect whether a small language
model's answer is *hallucinated* (fabricated) or *truthful* using the model's
own internal representations (hidden states).

> **Applicant: Tatevik Ter-Hovhannisyan.** The full write-up of the final approach, every
> phase tried (including failed attempts), and the contribution split between the applicant
> and the Claude Code assistant is in [SOLUTION.md](SOLUTION.md). The rest of this README
> documents the unchanged competition scaffolding plus our additions to it.
>
> **Headline result:** canonical `predictions.csv` is a majority-vote ensemble of 5
> probes (1× XGBoost + 4× MLP variants); local 5-fold test_acc of the strongest
> single well-calibrated component is **71.26%** (74.43% AUROC), best single component
> (XGBoost+features) is **74.16%**. Expected leaderboard accuracy: **72-74%**.
>
> **All runs tracked on Weights & Biases:** [wandb.ai/tatevik-th/smiles-2026-hallucination](https://wandb.ai/tatevik-th/smiles-2026-hallucination?nw=nwusertatevikt)

## Overview

Large (and small) language models sometimes *hallucinate* — they generate
plausible-sounding text that is factually incorrect.  This competition asks you
to build a **lightweight binary classifier** (called a *probe*) that reads the
model's internal hidden states and predicts whether a given response is
truthful (`label = 0`) or hallucinated (`label = 1`).

The language model used throughout is **[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)** — a
decoder-only causal transformer with 24 layers and a hidden dimension of 896.
It fits comfortably on a free Google Colab T4 GPU.

**Primary ranking metric:** Accuracy on the held-out `test.csv`.

## Repository Structure

```
SMILES-HALLUCINATION-DETECTION/
├── data/
│   ├── dataset.csv         # Labelled training data (prompt, response, label)
│   └── test.csv            # Unlabelled competition test set
│
├── solution.py             # Main script — extracts features, evaluates, writes predictions.csv
├── predictions.csv         # 5-model majority-vote ensemble, the submitted file
├── results.json            # 5-fold metrics from the best single well-calibrated config
├── SOLUTION.md             # Full applicant write-up (approach, phases, ablations, attribution)
│
│   ── Editable competition files (modified by the applicant) ──────────
├── aggregation.py          # Layer/token aggregation + geometric/attention/perplexity/heuristic feature extractors
├── probe.py                # HallucinationProbe — supports MLP (torch) and XGBoost backends; accuracy-mode threshold tuner
├── splitting.py            # Stratified single-split and 5-fold strategies; SPLIT_SEED env var
│
│   ── Fixed infrastructure (unchanged) ─────────────────────────────────
├── model.py                # Loads Qwen2.5-0.5B with output_hidden_states=True
├── evaluate.py             # Evaluation loop, metrics, summary table, JSON output
│
│   ── Applicant tooling (not part of the competition contract) ─────────
├── run_sweep.sh            # Env-var sweep launcher with per-run archiving (ONLY=N,M, START=N, SUFFIX="..." selectors)
├── compare_runs.py         # Cross-run leaderboard table (sort by test_acc / test_auroc / val_*)
├── error_analysis.py       # Per-sample misclassification dump on the canonical 5-fold config
├── embeddings_audit.py     # Distribution-shift check: dataset splits ↔ data/test.csv
├── slice_similarity.py     # Per-fold test-slice similarity to data/test.csv (drives weighted_metrics.py)
├── weighted_metrics.py     # Trust-weighted leaderboard estimates per run
├── ensemble_predictions.py # Majority-vote ensemble across multiple predictions.csv files
│
├── logs/
│   ├── runs/{run_name}/    # Per-run archive: results.json, predictions.csv, stdout.log
│   └── audits/             # embeddings_audit.json, slice_similarity.json, error_analysis.json, weighted_metrics.json
│
├── requirements.txt        # Python dependencies (+ xgboost added for the tree-based probe family)
├── wandb_utils.py          # Thin wandb integration used by solution.py
├── .env / .env.example     # WANDB_API_KEY etc. (.env is git-ignored)
└── LICENSE
```


## Quick Start

### Reproduce the submission exactly

The submitted `predictions.csv` is a 5-model majority-vote ensemble. Reproduce in three steps:

```bash
# 1) Pre-compute SelfCheckGPT and head-attribution features (one-time).
python selfcheck_features.py   # ~15 min, writes logs/audits/selfcheck_features.npz
python head_attribution.py     # ~30 s, writes logs/audits/head_attribution_*.{npy,npz,json}

# 2) Generate the 5 ensemble inputs.
ONLY="57,58,46,5" SUFFIX="-clean" ./run_sweep.sh   # 4 of the 5 inputs (xgb-everything-5fold, abl-no-perplexity, manifold-5fold, split-5fold-baseline)
ONLY="55" SUFFIX="" ./run_sweep.sh                  # the 5th: xgb-selfcheck-5fold

# 3) Build the ensemble + restore the canonical submission artifacts.
python ensemble_predictions.py \
  --runs xgb-everything-5fold-clean abl-no-perplexity-clean \
         xgb-selfcheck-5fold manifold-5fold-clean final-submission \
  --out logs/runs/ensemble-V4-clean/predictions.csv
./lock_canonical.sh   # copies ensemble-V4-clean/predictions.csv → predictions.csv, and xgb-everything-5fold-clean/results.json → results.json
```

⚠️ **Note:** any individual `python solution.py` run will overwrite `predictions.csv` and `results.json` in the repo root. After a sweep, always run `./lock_canonical.sh` to restore the submission artifacts from the per-run archives under `logs/runs/`.

### Run a single probe variant (the original competition pipeline)

```bash
git clone https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
python solution.py               # last_token + MLP + single stratified split + threshold-tuned final probe
```

To skip wandb tracking on any invocation, prepend `WANDB_MODE=disabled`.

### Hyperparameter knobs (env vars)

Every sweepable setting is an env var read by `solution.py`, `aggregation.py`, `probe.py`, or `splitting.py`. Set on the command line to override defaults:

| Env var | Default | Values |
|---|---|---|
| `AGG_STRATEGY` | `last_token` | `last_token`, `mean_pool`, `last4_concat`, `meanmaxlast` |
| `AGG_LAYER` | `-1` | int in `[-25, 24]` — Qwen2.5-0.5B layer index (24 = final) |
| `PROBE_ARCH` | `mlp_1h_256` | `mlp_1h_256`, `mlp_deep`, `linear` |
| `PROBE_FAMILY` | `torch` | `torch`, `xgboost` |
| `PROBE_WEIGHT_DECAY` | `0` | float L2 strength for Adam |
| `SPLIT_STRATEGY` | `single` | `single`, `5fold` |
| `SPLIT_SEED` | `42` | int |
| `USE_GEOMETRIC` | `0` | `0`/`1` |
| `USE_ATTENTION` | `0` | `0`/`1` (forces eager attention) |
| `USE_PERPLEXITY` | `0` | `0`/`1` |
| `USE_HEURISTIC` | `0` | `0`/`1` |
| `TEXT_MODE` | `prompt_response` | `prompt_response`, `prompt_only`, `response_only` |

## Dataset

`data/dataset.csv` contains 689 labelled samples with three columns:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Full ChatML-formatted conversation context fed to Qwen |
| `response` | str | The model's generated response |
| `label` | float | `1.0` = hallucinated · `0.0` = truthful |

The `prompt` uses the **ChatML** template built into Qwen models:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Given the context, answer the question …<|im_end|>
<|im_start|>assistant
```


`data/test.csv` is structured identically but the `label` column is null - these are the samples you submit predictions for via a `predictions.csv` generated file.


## What You Implement

You are expected to edit **three files**:  
- `aggregation.py`
- `probe.py`
- `splitting.py`

The rest of the codebase shall remain untouched.

**Feature Engineering & Dimensionality Reduction**: Applicants are encouraged to experiment with adding hand-crafted features during the aggregation step, drawing on geometrical or topological methods to enrich the representation of probe outputs. Additionally, you may apply dimensionality reduction techniques within probe.py to compress or refine the feature space. 

## Evaluation

For each fold `evaluate.py` reports four numbers:

| # | Checkpoint | Metrics |
|---|-----------|---------|
| 1 | Majority-class baseline | Accuracy, F1 |
| 2 | `HallucinationProbe` on **training** split | Accuracy, F1, AUROC |
| 3 | `HallucinationProbe` on **validation** split | Accuracy, F1, AUROC |
| 4 | `HallucinationProbe` on **test** split | Accuracy, F1, AUROC |

**Accuracy on the `test.csv` is the primary competition metric.**

Results are averaged across folds (if using k-fold) and saved to
`results.json`.


# What is expected from the applicant of SMILES-2026 ?

**Q1:** What must the applicant submit in the application form ?<br>
**A1:** Submit: 
1. A link to your Github repository
2. A link to your `predictions.csv` publicly available file on some cloud storage

**Q2:** What the applicants must include in the repository ?<br>
**A2:** Your repository must contain: 
1. `results.json` - produced by the official `solution.py`
2. Report file in Markdown format `SOLUTION.md`. 

**Q3:** Report requirements (`SOLUTION.md`)<br>
**A3:** Your report must include:<br>
- Reproducibility instructions: exact commands to run your solution and acquire the same `predictions.csv`, required environment (if any), any important implementation details needed to reproduce your result.
- Final solution description: What components you modified ? What your final approach is ? Why you made these choices ? What contributed most to improving the metric ?
- Experiments and failed attempts: What ideas you tried but did not include in the final solution ? Why they did not work or were discarded ?

**Q4:** Reproducibility<br>
**A4:** The repository must be self-contained and runnable with the provided `solution.py` file. Your solution must not require changes to the fixed infrastructure files. Running `solution.py` must generate your submitted `predictions.csv`.
