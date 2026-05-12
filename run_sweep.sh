#!/usr/bin/env bash
# Five tagged baseline runs over aggregation / probe / split knobs.
# Invoke from smiles/repo/. Each run lands in wandb with its own name + tags.
#
# Override defaults at the command line, e.g.:
#   ONLY=1 ./run_sweep.sh         # only run the first row
#   PYTHON=python3 ./run_sweep.sh # custom interpreter

set -euo pipefail

PYTHON="${PYTHON:-/home/tterhovhanni/Desktop/smiles/.venv/bin/python}"
ONLY="${ONLY:-}"
START="${START:-1}"
SUFFIX="${SUFFIX:-}"

run() {
    local idx="$1"; shift
    local name="$1${SUFFIX}"; shift
    local tags="$1"; shift
    if [[ -n "$ONLY" ]]; then
        local match=0
        for o in ${ONLY//,/ }; do
            if [[ "$o" == "$idx" ]]; then match=1; break; fi
        done
        if (( match == 0 )); then
            echo "── skipping [$idx] $name (ONLY=$ONLY)"
            return
        fi
    fi
    if (( idx < START )); then
        echo "── skipping [$idx] $name (START=$START)"
        return
    fi
    local outdir="logs/runs/$name"
    mkdir -p "$outdir"
    # Wipe stale artifacts so a crashed run can't archive a previous one's output.
    rm -f results.json predictions.csv
    echo "════════════════════════════════════════════════════════════"
    echo " [$idx] $name  tags=$tags  → $outdir/"
    echo "════════════════════════════════════════════════════════════"
    env "$@" WANDB_RUN_NAME="$name" WANDB_TAGS="$tags" "$PYTHON" solution.py 2>&1 | tee "$outdir/stdout.log"
    [[ -f results.json ]]    && cp results.json    "$outdir/results.json"
    [[ -f predictions.csv ]] && cp predictions.csv "$outdir/predictions.csv"
}

run 1 "agg-last-token"   "agg"             AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single
run 2 "agg-mean-pool"    "agg"             AGG_STRATEGY=mean_pool    PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single
run 3 "agg-last4-layers" "agg,multi-layer" AGG_STRATEGY=last4_concat PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single
run 4 "probe-deep-mlp"   "probe"           AGG_STRATEGY=last_token   PROBE_ARCH=mlp_deep   SPLIT_STRATEGY=single
run 5 "split-5fold"      "cv"              AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold
# ── Phase 4: linear probes + L2 regularization (builds on phase-2 acc-thresh winner) ──
run 6 "probe-linear"     "probe,linear"    AGG_STRATEGY=last_token   PROBE_ARCH=linear     SPLIT_STRATEGY=single
run 7 "probe-linear-l2"  "probe,linear"    AGG_STRATEGY=last_token   PROBE_ARCH=linear     SPLIT_STRATEGY=single PROBE_WEIGHT_DECAY=1e-3
run 8 "probe-mlp-l2"     "probe,l2"        AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single PROBE_WEIGHT_DECAY=1e-3
# ── Phase 5: geometric features (per-layer norms + inter-layer cosine drift + seq_len) ──
run 9  "geom-features"    "geom"           AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single USE_GEOMETRIC=1
run 10 "geom-features-l2" "geom,l2"        AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single USE_GEOMETRIC=1 PROBE_WEIGHT_DECAY=1e-2
# ── Phase 6: multi-seed final estimate on the winning config ──
run 11 "seed-0"           "multi-seed"     AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single SPLIT_SEED=0
run 12 "seed-1"           "multi-seed"     AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single SPLIT_SEED=1
run 13 "seed-2"           "multi-seed"     AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single SPLIT_SEED=2
run 14 "seed-3"           "multi-seed"     AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single SPLIT_SEED=3
run 15 "seed-4"           "multi-seed"     AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single SPLIT_SEED=4
# ── Phase 7: abstention-style probing — does the prompt alone predict hallucination? ──
run 16 "text-prompt-only"  "text"          AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single TEXT_MODE=prompt_only
run 17 "text-response-only" "text"         AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single TEXT_MODE=response_only
# ── Phase 8: mid-layer probing (Burns et al., your intern's finding) ──
# Qwen2.5-0.5B has 24 transformer layers; hidden_states indexed 0 (embed) .. 24 (final).
run 18 "layer-08"  "layer-scan"             AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single AGG_LAYER=8
run 19 "layer-12"  "layer-scan"             AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single AGG_LAYER=12
run 20 "layer-14"  "layer-scan"             AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single AGG_LAYER=14
run 21 "layer-16"  "layer-scan"             AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single AGG_LAYER=16
run 22 "layer-18"  "layer-scan"             AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single AGG_LAYER=18
run 23 "layer-20"  "layer-scan"             AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single AGG_LAYER=20
# ── Phase 9: multi-token aggregation (mean+max+last concat over real tokens) ──
run 24 "agg-meanmaxlast" "agg,multi-token"  AGG_STRATEGY=meanmaxlast  PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single
# ── Phase 8.5: canonical final submission (last_token + 5fold + threshold-tuned final probe) ──
run 25 "final-submission" "final"           AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold
# ── Phase 9 + 5-fold combo ──
run 26 "meanmaxlast-5fold" "agg,multi-token,cv"  AGG_STRATEGY=meanmaxlast PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold
# ── Phase 10: attention features (per-layer entropy / self-attn / top-3 concentration) ──
run 27 "attn-last-token"   "attention"     AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single USE_ATTENTION=1
run 28 "attn-meanmaxlast"  "attention,multi-token" AGG_STRATEGY=meanmaxlast PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single USE_ATTENTION=1
run 29 "attn-last-5fold"   "attention,cv"  AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold USE_ATTENTION=1
# ── Phase 11: response NLL / perplexity features (6 extra dims) ──
run 30 "ppl-last-token"    "perplexity"    AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single USE_PERPLEXITY=1
run 31 "ppl-last-5fold"    "perplexity,cv" AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold USE_PERPLEXITY=1
# ── Phase 13: heuristic text features (response length, hedge-words, prompt-overlap) ──
run 32 "heur-last-token"   "heuristic"     AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single USE_HEURISTIC=1
run 33 "heur-last-5fold"   "heuristic,cv"  AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold USE_HEURISTIC=1
# ── Phase 14: ALL THE THINGS — last_token + perplexity + heuristic + 5-fold ──
run 34 "all-features-5fold" "all"          AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold USE_PERPLEXITY=1 USE_HEURISTIC=1
# ── Phase 12: XGBoost probe family ──
run 35 "xgb-last-token"    "xgboost"       AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single PROBE_FAMILY=xgboost
run 36 "xgb-last-5fold"    "xgboost,cv"    AGG_STRATEGY=last_token   PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold PROBE_FAMILY=xgboost
run 37 "xgb-all-features-5fold" "xgboost,all,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold PROBE_FAMILY=xgboost USE_PERPLEXITY=1 USE_HEURISTIC=1
# ── Phase 15: logit-lens features (per-layer LM-head projections; "with norm") ──
run 38 "ll-last-token"     "logit-lens"    AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single USE_LOGIT_LENS=1
run 39 "ll-last-5fold"     "logit-lens,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold USE_LOGIT_LENS=1
run 40 "ll-all-features-5fold" "logit-lens,all,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold USE_LOGIT_LENS=1 USE_PERPLEXITY=1 USE_HEURISTIC=1
run 41 "xgb-ll-all-5fold"  "xgboost,logit-lens,all,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold PROBE_FAMILY=xgboost USE_LOGIT_LENS=1 USE_PERPLEXITY=1 USE_HEURISTIC=1
# ── Phase 13: KNN-OOD distance features (manifold/density signals) ──
run 42 "knn-last-token"    "knn-ood"       AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=single USE_KNN_OOD=1
run 43 "knn-last-5fold"    "knn-ood,cv"    AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold USE_KNN_OOD=1
run 44 "knn-all-features-5fold" "knn-ood,all,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold USE_KNN_OOD=1 USE_PERPLEXITY=1 USE_HEURISTIC=1
run 45 "xgb-knn-all-5fold" "xgboost,knn-ood,all,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold PROBE_FAMILY=xgboost USE_KNN_OOD=1 USE_PERPLEXITY=1 USE_HEURISTIC=1
# ── Phase 14+15: manifold features (LID + per-class Mahalanobis) ──
run 46 "manifold-5fold"    "manifold,cv"   AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold USE_MANIFOLD=1
run 47 "xgb-manifold-all-5fold" "xgboost,manifold,all,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold PROBE_FAMILY=xgboost USE_MANIFOLD=1 USE_PERPLEXITY=1 USE_HEURISTIC=1
run 48 "xgb-knn-manifold-all-5fold" "xgboost,knn-ood,manifold,all,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold PROBE_FAMILY=xgboost USE_KNN_OOD=1 USE_MANIFOLD=1 USE_PERPLEXITY=1 USE_HEURISTIC=1
# ── Phase 3: cross-attention from response to prompt span ──
run 49 "attn2prompt-5fold"     "attention,attn2prompt,cv"      AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold USE_ATTENTION=1 USE_ATTENTION_TO_PROMPT=1
run 50 "xgb-attn2prompt-all-5fold" "xgboost,attention,attn2prompt,all,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold PROBE_FAMILY=xgboost USE_ATTENTION=1 USE_ATTENTION_TO_PROMPT=1 USE_PERPLEXITY=1 USE_HEURISTIC=1
run 51 "xgb-manifold-attn2prompt-all-5fold" "xgboost,manifold,attention,attn2prompt,all,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold PROBE_FAMILY=xgboost USE_MANIFOLD=1 USE_ATTENTION=1 USE_ATTENTION_TO_PROMPT=1 USE_PERPLEXITY=1 USE_HEURISTIC=1
# ── Phase 5: per-head attention probing (336 per-(layer, head) features) ──
run 52 "xgb-perhead-attn-5fold" "xgboost,per-head,attention,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold PROBE_FAMILY=xgboost USE_ATTENTION=1 USE_PER_HEAD_ATTN=1
run 53 "xgb-manifold-perhead-all-5fold" "xgboost,manifold,per-head,attention,all,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold PROBE_FAMILY=xgboost USE_MANIFOLD=1 USE_ATTENTION=1 USE_PER_HEAD_ATTN=1 USE_PERPLEXITY=1 USE_HEURISTIC=1
run 54 "xgb-manifold-perhead-attn2prompt-all-5fold" "xgboost,manifold,per-head,attn2prompt,all,cv" AGG_STRATEGY=last_token PROBE_ARCH=mlp_1h_256 SPLIT_STRATEGY=5fold PROBE_FAMILY=xgboost USE_MANIFOLD=1 USE_ATTENTION=1 USE_ATTENTION_TO_PROMPT=1 USE_PER_HEAD_ATTN=1 USE_PERPLEXITY=1 USE_HEURISTIC=1
