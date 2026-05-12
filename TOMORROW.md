# Tomorrow's experiments — bold, architecturally-grounded

> Save for resumption. Picks up from where we left off on 2026-05-11.

## Where we are tonight

- **Submission pushed.** [github.com/tatevik-t/smiles-2026-submission](https://github.com/tatevik-t/smiles-2026-submission) — commit `4a5dd2d`.
- **Canonical `predictions.csv`** is a 5-model majority-vote ensemble (XGBoost + 4× MLP variants), 80/20 hallucinated.
- **Local best single-model 5-fold:** `xgb-all-features-5fold-v2` at 74.16% test_acc / 74.09% AUROC, but with a calibration-skewed 96/4 predictions.csv.
- **Best well-calibrated single-model:** `all-features-5fold` (MLP + perplexity + heuristic) at 71.26% test_acc / 74.43% AUROC (this is what `results.json` reports).
- **Expected leaderboard accuracy:** 72-74%.
- **Multi-seed std on test_acc:** ±2.76% — small movements within this band are noise; we need ≥3pt jumps to confidently say something helped.

## Goal for tomorrow

Push the leaderboard estimate to **75%+** with experiments that go beyond standard probing into **architecture-specific** and **manifold-geometric** signals. Failing that, generate insights for SOLUTION.md that make the work look strong even if the numbers don't move.

---

## Tier A — established prior art, high-leverage

### 1. Logit lens features (top architectural-payoff per minute)

**Hypothesis.** Truthful responses have *stable* next-token predictions across layers — the answer crystallizes early in the residual stream and persists. Hallucinated responses have *unstable* predictions — the late layers fabricate something the early layers didn't predict.

**Implementation.**
- For each response token at position `t`, take hidden state `h_l[t]` at every layer `l` (0..24).
- Project each through Qwen's LM head: `logits_l[t] = h_l[t] @ lm_head.weight.T`. Qwen has tied embeddings, so `lm_head.weight = embed_tokens.weight`.
- Per-sample features (summary stats, NOT per-layer × per-token to keep dim manageable):
  - **Layer-at-which-top-1-first-matches-actual-token**, mean and max across response tokens
  - **KL(layer_L_logits || layer_24_logits)** averaged across response tokens, for L ∈ {8, 12, 16, 20}
  - **NLL trajectory variance** across layers
  - **"Late commitment" indicator**: fraction of response tokens where the top-1 only appears in the last 4 layers
- ~10 dims total.

**Cost.** ~25 layers × matmul `(seq_len, hidden_dim) @ (hidden_dim, vocab_size)`. Per sample: ~1.7B FLOPs ≈ 10ms. Whole dataset: ~7s.

**Risk.** Established on 7B+ models (`logit-lens` blog, Belrose et al. `tuned-lens` 2023). At 0.5B / 24 layers the trajectory may be too short. **+1-4pt likely, possibly nothing.**

**Where to put it.** New `extract_logit_lens_features(hidden_states, input_ids, lm_head_weight, prompt_token_count)` in `aggregation.py`. New `USE_LOGIT_LENS` env var in `solution.py`. Pull `lm_head_weight` from `model.lm_head.weight` (or `model.get_output_embeddings().weight`) and pass via the extraction loop.

### 2. SelfCheckGPT proper (5× generation, semantic divergence)

**Hypothesis.** Sample Qwen's response 5× at temperature=0.7 for each prompt. Hallucinated prompts produce high inter-sample semantic divergence (model has no anchor); truthful prompts produce consistent samples.

**Implementation.**
- `model.generate(prompt, do_sample=True, temperature=0.7, num_return_sequences=5, max_new_tokens=60)` per sample.
- For each sample's 5 alternatives, compute:
  - **Mean pairwise BLEU** (`sacrebleu` or `nltk.bleu`)
  - **Mean pairwise cosine** of last-token embedding under same Qwen (re-feed each alternative through extraction pipeline)
  - **Mean pairwise n-gram overlap** (Jaccard on 3-grams)
- 3-5 features.

**Cost.** Qwen2.5-0.5B generates ~80 tok/s on the 5880; mean response ~30 tokens; 5× per sample × 789 samples ≈ 25 min. Heaviest single experiment.

**Risk.** De-facto SOTA for hallucination detection in 2024-2025. **+2-5pt likely.** Lower variance than logit lens.

**Where to put it.** Separate script `selfcheck_features.py` (computes features once, saves npy) — don't bake into `solution.py` since the generation is expensive and not part of the per-run feature loop.

### 3. Cross-attention from response to prompt span

**Hypothesis.** When answering correctly, the model attends strongly to specific prompt tokens (relevant context). When fabricating, attention is diffuse or fixates on wrong tokens.

**Implementation.**
- **First: prompt-boundary tracking.** `tokenizer.encode(prompt, add_special_tokens=False)` → `prompt_len_tokens`. Store per sample.
- For each (layer, head) at each response token position, compute fraction of attention mass landing inside the prompt span (positions `[0, prompt_len_tokens)`).
- Aggregate: per-layer mean of response→prompt attention mass (24 dims), summary stats (8 dims). ~32 features total.

**Cost.** Free — we already extract attentions when `USE_ATTENTION=1`. Just a different aggregation.

**Risk.** The current `attn-last-5fold` features (entropy + top-3 + self-attn) were a wash. The response→prompt mass is the more interpretable / load-bearing signal. **+1-3pt likely.** Plus, the prompt-boundary tracking is infrastructure that #1, #4, and the response-only multi-token followup need.

**Where to put it.** Extend `extract_attention_features` to accept `prompt_token_count` and return additional response→prompt features. New env var `USE_ATTENTION_TO_PROMPT=1` to enable.

---

## Tier B — riskier, less established

### 4. NLL trajectory shape features (extended perplexity)

**Hypothesis.** Hallucinated responses often have a "low-NLL prefix" (model on safe ground) → "NLL spike" (fabrication moment) → "high-NLL tail". Our current per-token NLL summary (mean/max/std) collapses this temporal structure.

**Features added** (on top of existing 6 perplexity dims):
- Position of max-NLL within the response (normalized to [0, 1])
- 1st-order autocorrelation of NLL across response tokens
- Linear-fit slope of NLL vs position
- Kurtosis of NLL distribution
- "NLL ratio prefix/suffix" — mean of first-third vs mean of last-third

5-6 extra dims, drop-in extension of `extract_perplexity_features`.

**Cost.** Trivial.

**Risk.** Hypothesis depends on response length. Mean response ~30 tokens — trajectory may be too short for shape features. **+0-2pt likely.**

### 5. Per-head probing — find the "hallucination heads"

**Hypothesis.** A handful of the 24×14 = 336 attention heads carry most of the hallucination signal (analogous to "induction heads"). Identify them, probe just their attention patterns.

**Implementation.**
- For each head, compute a 1-dim summary stat per sample (e.g., attention-to-last-prompt-token at last-response-token).
- Mutual-information rank of each head against the label.
- Pick top-K=30 heads, use their per-sample stats as a 30-dim feature vector.

**Cost.** Cheap once attentions are extracted. MI computation: ~5s.

**Risk.** Probably won't outperform the global attention features by much, but **interpretable and great for SOLUTION.md** ("we localized hallucination signal to specific heads in layers X-Y"). **+0-2pt.**

### 6. Layer-wise probe ensemble (rethink)

**Hypothesis.** Phase 8 showed final layer wins on its own, but a *combination* of per-layer probes might beat any single layer.

**Implementation.**
- Train 25 separate `HallucinationProbe(AGG_LAYER=l)` instances, one per layer.
- Average their predicted probabilities → final prediction. OR: use per-layer probabilities as features for a meta-probe (XGBoost on 25-dim probability vector).

**Cost.** 25× the training compute (still fast). Per-sample feature extraction needs all 25 layers' last-token states — already extracted.

**Risk.** **+1-2pt likely** but partly redundant with the existing ensemble.

---

## Tier C — research-grade, possibly nothing

### 7. Sparse autoencoder dictionary on the residual stream

**Hypothesis.** Decompose the 896-dim hidden state into ~16K sparse interpretable features (Anthropic-style dictionary learning). Probe on the sparse features instead of the raw hidden state.

**Implementation.** Train a SAE with L1 sparsity on the residual stream of the final layer. Dict size 16K, ~50 active features per sample.

**Cost.** SAE training ~5-30 min. Then probe on sparse activations.

**Risk.** SAEs need significantly more data than 689 samples to learn meaningful features. **Likely fails.** Reserve for the stretch slot.

### 8. Activation patching for causal feature identification

**Hypothesis.** Pick a hallucinated and a truthful sample. At layer L, swap their hidden states. If probe prediction flips, layer L causally encodes the hallucination signal.

**Implementation.** Hook into `model.model.layers[l].forward` to allow per-layer residual injection. Run paired swaps.

**Cost.** Per-pair forward × pairs × layers. ~10-20 min for a careful sweep.

**Risk.** **Diagnostic, not for accuracy** — tells you *where* the signal lives, not how to extract better features. Worth doing for SOLUTION.md gravitas.

### 9. Verification reframing — ask Qwen "is this correct?"

**Hypothesis.** Reformulate as instruction-following: prompt Qwen with "Given the context, is the following answer correct? \<prompt\> \<response\> Answer (yes/no):" and read the logit ratio at the next position.

**Implementation.** One forward pass per sample with reformulated prompt. Features: logit difference between "yes" and "no" tokens.

**Cost.** One extra forward pass per sample. ~7s.

**Risk.** Qwen2.5-0.5B is small; may not understand verification reframing. Bigger models do. **+0-3pt likely.**

---

## Tier D — loss-gradient features (the model's own training signal)

### 10. Per-layer gradient norm of response NLL

**Hypothesis.** Truthful responses produce small per-layer gradients (model already fits them); hallucinated responses produce large gradients (model is "surprised").

**Implementation.**
```python
for sample_i in range(N):
    model.zero_grad()
    input_ids = ...  # single sample, batch dim 1
    logits = model(input_ids).logits
    loss = compute_response_nll(logits, input_ids, prompt_token_count[i])
    loss.backward()
    grad_norms = []
    for layer in model.model.layers:
        g_sq = sum((p.grad.norm()**2 for p in layer.parameters() if p.grad is not None))
        grad_norms.append(g_sq.sqrt().item())
    features[i] = grad_norms + [mean(grad_norms), max(grad_norms)]
```
- 24 layer norms + 2 summary = **26 dims**.
- Per-sample backward pass at batch=1; ~20-30 sec total for 789 samples.

**Risk.** Established in vision (Grad-CAM-style); less in LLMs. **+1-3pt potential, high implementation overhead** because of per-sample (not batched) backward.

**Where to put it.** Separate script `gradient_features.py` (saves npy), then load in solution.py as additional features when `USE_GRAD_FEATURES=1`.

### 11. Input-embedding gradient saliency

**Hypothesis.** ∂NLL/∂embed_at_position_t for each token is a saliency map. Tokens that the model "depended on" should differ between truthful and hallucinated.

**Implementation.** Set `embed_tokens(input_ids).requires_grad_(True)` after manually invoking the embedding layer; forward; backward. Per-token gradient L2 norms.

**Features.** Aggregate over response span — mean, max, std, position of max gradient. **5 dims.**

**Cost.** Similar to #10 (single backward per sample).

**Risk.** **+0-2pt.**

### 12. Layer-conditioned gradient *direction*

**Hypothesis.** ∂P(hallucinated)/∂h_l for each layer l. Cosine similarity between this gradient direction and a reference hallucinated-direction (mean across training set's hallucinated samples) is a feature.

**Implementation.** Builds on the trained probe — gradient is taken through the probe + last-layer hidden state, treating earlier layers as frozen.

**Risk.** **Diagnostic, not accuracy.** Useful for SOLUTION.md.

---

## Tier E — manifold / OOD-detection features

### 13. k-NN distance features in hidden-state space

**Hypothesis.** Hallucinated responses might live in less dense regions of the embedding space, or have unusual NN profiles relative to the two classes.

**Features per sample.**
- Distance to 5th-nearest training neighbor (class-agnostic)
- Distance to 5th-nearest *truthful* training neighbor
- Distance to 5th-nearest *hallucinated* training neighbor
- Ratio: NN-truthful / NN-hallucinated
- Mean cosine to 10 nearest neighbors

**6 dims, CPU, ~0.5s** on 689 samples.

**Risk.** Well-established for OOD detection (KNN-OOD, Sun et al. ICML 2022). KNN has a different inductive bias from our parametric probes — should compose well. **+1-3pt likely.** Best risk-adjusted bet in Tier E.

**Where to put it.** Compute once in solution.py (after main feature extraction), append to feature vector. Env var `USE_KNN_OOD=1`.

### 14. Local intrinsic dimensionality (TwoNN)

**Hypothesis.** Samples in "thin" embedding regions (low LID) are on well-trodden manifold (truthful pattern); "thick" regions (high LID) are off-manifold.

**Features.** TwoNN LID estimate (Facco et al. 2017) — uses ratio of 1st-NN to 2nd-NN distances per sample. **2 dims (raw LID + log LID).**

**Cost.** Cheap.

**Risk.** Sample-size-sensitive at 689. **+0-2pt.**

### 15. Per-class Mahalanobis distance

**Hypothesis.** Fit a multivariate Gaussian to each class in training embedding space. Mahalanobis distance from each test sample to each class's Gaussian is a feature.

**Features.** Distance to truthful Gaussian, distance to hallucinated Gaussian, ratio, difference. **4 dims.**

**Risk.** Essentially LDA-with-a-different-framing — partly redundant with our probe. **+0-1pt.**

### 16. Persistent-homology features (the wildest)

**Hypothesis.** Topological features of the dataset's embedding cloud. Per-sample summary stats from the persistence diagram.

**Implementation.** `ripser` / `giotto-tda` on PCA-reduced embeddings.

**Cost.** PH on ~700 points in 50-D ~ 1 min. Per-sample summaries are tricky — TDA usually gives dataset-level features.

**Risk.** **Most likely to fail for accuracy purposes.** Save for "I want a beautiful story for the report" tier. **+0-1pt.**

---

## Proposed schedule (4-5 hours, prioritized)

| Slot | Phase | Time | Why now |
|---|---|---|---|
| 9:00–10:30 | **#1 Logit lens** | 90 min | Top architectural-payoff-per-minute, cheap compute |
| 10:30–11:00 | **#13 KNN-OOD** | 30 min | Cheap, well-established, complementary inductive bias |
| 11:00–12:00 | **#3 Cross-attention to prompt** | 60 min | Includes prompt-boundary tracking infra reusable by #1, #4 |
| Lunch | **#2 SelfCheckGPT** (background) | runs ~25 min | Kick off the generation loop before lunch |
| 13:00–14:00 | **#10 Per-layer gradient norm** | 60 min | The bold per-sample-backward run |
| 14:00–14:45 | **#14 LID + #15 Mahalanobis** | 45 min | Bundle into one `manifold_features.py` |
| 14:45–15:30 | **#4 NLL trajectory + #5 per-head probing + #6 layer ensemble** | 45 min | Cheap follow-ups; bundle |
| 15:30–17:00 | **Re-ensemble + SOLUTION.md** | 90 min | Train final ensemble including every config that beat baseline. Write up. |
| Stretch | **#7 SAE** or **#16 persistent homology** | 60-90 min | Pick ONE. High-narrative-value, low-leaderboard-EV. |

## Risk budget

- **Do not bet more than one slot on SAE or PH.** Both are likely to consume hours and yield nothing.
- **Logit lens, KNN-OOD, gradient norms, cross-attention-to-prompt are the four highest-leverage independent signals.** If even two of them give +1-2pt each, the ensemble will land ≥75%.
- **The submission is already pushed and viable.** Tomorrow's work is to push higher, not to recover from a failure mode.

## Highest-leverage single bet

**Logit lens + per-layer gradient norm + KNN-OOD** are three completely independent signals, each architecturally principled in a different way:
- Logit lens → model's *internal predictions*
- Gradient norm → model's *training-signal sensitivity*
- KNN-OOD → *geometric outlier-ness* in embedding space

Stacking all three onto the current `all-features-5fold` config could plausibly push to 76-78% local 5-fold test_acc. **Big if it works.** If it doesn't, we still have a strong submission (72-74% expected leaderboard) and a great SOLUTION.md narrative.

## Resumption checklist

When you come back tomorrow:
1. `cd /home/tterhovhanni/Desktop/smiles/repo`
2. `source /home/tterhovhanni/Desktop/smiles/.venv/bin/activate`
3. Verify GPU: `nvidia-smi || python -c "import torch; print(torch.cuda.is_available())"` (NVML may still show a mismatch — that's OK, CUDA itself works)
4. Confirm git state: `git status` (should be clean), `git remote -v` (should point at `tatevik-t/smiles-2026-submission`)
5. Read this file. Start with phase 1.

Good luck.
