#!/usr/bin/env bash
# Restores the canonical submission artifacts from their authoritative
# locations under logs/runs/. Run this after any sweep that may have
# overwritten predictions.csv / results.json in the repo root.

set -euo pipefail

CANONICAL_PREDS_RUN="ensemble-V4-clean"
CANONICAL_RESULTS_RUN="xgb-everything-5fold-clean"

cp "logs/runs/${CANONICAL_PREDS_RUN}/predictions.csv"  predictions.csv
cp "logs/runs/${CANONICAL_RESULTS_RUN}/results.json"   results.json

echo "── canonical restored ──"
echo "  predictions.csv ← logs/runs/${CANONICAL_PREDS_RUN}/predictions.csv"
echo "  results.json    ← logs/runs/${CANONICAL_RESULTS_RUN}/results.json"
python -c "
import pandas as pd, json
p = pd.read_csv('predictions.csv')
r = json.load(open('results.json'))
print(f'  → predictions: {p[\"label\"].sum()}/{len(p)} hallucinated ({p[\"label\"].mean()*100:.0f}%)')
print(f'  → results:     test_acc={r[\"avg_test_accuracy\"]*100:.2f}% AUROC={r[\"avg_test_auroc\"]*100:.2f}%')
"
