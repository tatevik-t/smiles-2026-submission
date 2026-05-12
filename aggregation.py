"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import os

import torch

# Sweep knob. Valid values: "last_token" (default), "mean_pool", "last4_concat",
# "meanmaxlast" (concat of mean / max / last over all real tokens — 3*hidden_dim).
AGG_STRATEGY = os.environ.get("AGG_STRATEGY", "last_token")
# Layer index for last_token / mean_pool. Default -1 = final transformer layer.
# Positive indices count from the embedding (0); negative from the end.
# For Qwen2.5-0.5B (24 layers + embedding), valid range is [-25, 24].
AGG_LAYER = int(os.environ.get("AGG_LAYER", "-1"))


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # attention_mask may live on CPU while hidden_states are on GPU.
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_positions[-1].item())

    if AGG_STRATEGY == "last_token":
        return hidden_states[AGG_LAYER][last_pos]

    if AGG_STRATEGY == "mean_pool":
        mask_f = attention_mask.float().unsqueeze(-1).to(hidden_states.device)
        layer = hidden_states[AGG_LAYER]                     # (seq_len, hidden_dim)
        return (layer * mask_f).sum(dim=0) / mask_f.sum().clamp(min=1.0)

    if AGG_STRATEGY == "last4_concat":
        return torch.cat([hidden_states[k][last_pos] for k in (-1, -2, -3, -4)], dim=0)

    if AGG_STRATEGY == "meanmaxlast":
        # Concat of mean, max, and last-token pooling over the AGG_LAYER layer.
        # Mean and max are computed over real (non-padding) tokens.
        layer = hidden_states[AGG_LAYER]                          # (seq_len, hidden_dim)
        mask_d = attention_mask.float().to(layer.device)          # (seq_len,)
        mask_f = mask_d.unsqueeze(-1)                             # (seq_len, 1)
        denom = mask_f.sum().clamp(min=1.0)
        mean_v = (layer * mask_f).sum(dim=0) / denom
        # For max, mask padding positions out with a very negative value.
        layer_for_max = layer.masked_fill(mask_f == 0, float("-inf"))
        max_v = layer_for_max.max(dim=0).values
        last_v = layer[last_pos]
        return torch.cat([mean_v, max_v, last_v], dim=0)

    raise ValueError(f"unknown AGG_STRATEGY={AGG_STRATEGY!r}")


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # Last real (non-padding) token position.
    last_pos = int(attention_mask.nonzero(as_tuple=False)[-1].item())

    # (n_layers, hidden_dim): last-token activations across every layer (incl. embedding).
    last_token_vecs = hidden_states[:, last_pos, :]

    # Per-layer L2 norm of the last-token activation (n_layers numbers).
    norms = torch.linalg.norm(last_token_vecs, dim=-1)

    # Cosine similarity between consecutive layers' last-token activations
    # (n_layers - 1 numbers). Captures representational drift across depth —
    # hallucinated answers tend to drift differently than truthful ones.
    cos_drift = torch.nn.functional.cosine_similarity(
        last_token_vecs[:-1], last_token_vecs[1:], dim=-1
    )

    # Sequence length (in real tokens), a single scalar.
    seq_len = attention_mask.sum().float().unsqueeze(0).to(norms.device)

    return torch.cat([norms, cos_drift, seq_len], dim=0)


def extract_attention_features(
    attentions: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_token_count: int | None = None,
) -> torch.Tensor:
    """Attention-based features for hallucination detection.

    Args:
        attentions:           Tensor of shape ``(n_layers, n_heads, seq_len, seq_len)``
                              for one sample.
        attention_mask:       1-D tensor of shape ``(seq_len,)`` with 1 for real
                              tokens and 0 for padding.
        prompt_token_count:   If provided (>0), also emit cross-attention-to-prompt
                              features: for each layer, the mean fraction of
                              attention from response tokens onto the prompt span,
                              plus summary stats across layers.

    Returns:
        1-D tensor. If ``prompt_token_count`` is None or 0: ``3 * n_layers``
        features (per-layer entropy, self-attn weight, top-3 concentration of
        the last-token attention distribution). Otherwise an additional
        ``n_layers + 5`` features appended.
    """
    last_pos = int(attention_mask.nonzero(as_tuple=False)[-1].item())

    attn_from_last = attentions[:, :, last_pos, :].mean(dim=1)            # (n_layers, seq_len)
    real_mask = attention_mask.float().to(attn_from_last.device)
    attn_masked = attn_from_last * real_mask.unsqueeze(0)
    attn_norm = attn_masked / attn_masked.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    log_p = torch.log(attn_norm.clamp(min=1e-9))
    entropy = -(attn_norm * log_p).sum(dim=-1)
    self_attn = attn_norm[:, last_pos]
    top3 = attn_norm.topk(min(3, attn_norm.size(-1)), dim=-1).values.sum(dim=-1)

    base = torch.cat([entropy, self_attn, top3], dim=0)

    if prompt_token_count is None or prompt_token_count <= 0:
        return base

    # ── cross-attention-to-prompt features ─────────────────────────────────
    n_layers, seq_len = attn_from_last.shape
    real_length = last_pos + 1
    if prompt_token_count >= real_length:
        # No response tokens at all; emit zeros for the extra dims.
        zero_extra = torch.zeros(n_layers + 5, device=attn_from_last.device)
        return torch.cat([base, zero_extra], dim=0)

    # For each layer, compute the mean (over response tokens) of the fraction
    # of attention mass landing inside the prompt span. Use head-mean attn.
    attn_head_mean = attentions[:, :, :real_length, :real_length].mean(dim=1)  # (n_layers, real_length, real_length)
    # Renormalize per query token across the real key positions.
    attn_head_mean = attn_head_mean / attn_head_mean.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    response_qpositions = slice(prompt_token_count, real_length)
    # Fraction of attention to prompt span (key positions [0, prompt_token_count)).
    prompt_attn_mass = attn_head_mean[:, response_qpositions, :prompt_token_count].sum(dim=-1)  # (n_layers, n_resp)
    per_layer_mean = prompt_attn_mass.mean(dim=-1)                        # (n_layers,)

    extra = torch.cat([
        per_layer_mean,                                                   # (n_layers,)
        per_layer_mean.mean().unsqueeze(0),                               # mean across layers
        per_layer_mean.max().unsqueeze(0),
        per_layer_mean.std().unsqueeze(0) if n_layers > 1 else torch.zeros(1, device=base.device),
        per_layer_mean.argmax().float().unsqueeze(0),                     # layer at which prompt-attn peaks
        prompt_attn_mass[:, -1].mean().unsqueeze(0),                       # last-response-token's attn-to-prompt averaged over layers
    ], dim=0)

    return torch.cat([base, extra], dim=0)


def extract_per_head_attention_to_prompt(
    attentions: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_token_count: int,
) -> torch.Tensor:
    """Per-(layer, head) attention-to-prompt features (phase 5 — per-head probing).

    Don't collapse over heads — emit one feature per (layer, head) pair so a
    tree-based probe can find the specific heads that carry the hallucination
    signal (analogous to "induction heads" in mechanistic interpretability).

    Args:
        attentions:           (n_layers, n_heads, seq_len, seq_len).
        attention_mask:       (seq_len,) with 1 for real tokens, 0 for pad.
        prompt_token_count:   number of prompt tokens.

    Returns:
        1-D tensor of shape (n_layers * n_heads,) — per-head fraction of
        attention mass from response tokens to prompt span, averaged over
        response query positions. Zero-filled if no response tokens exist.
    """
    last_pos = int(attention_mask.nonzero(as_tuple=False)[-1].item())
    real_length = last_pos + 1
    n_layers, n_heads = attentions.shape[0], attentions.shape[1]
    if prompt_token_count <= 0 or prompt_token_count >= real_length:
        return torch.zeros(n_layers * n_heads, device=attentions.device)

    # (n_layers, n_heads, n_resp, real_length)
    attn = attentions[:, :, prompt_token_count:real_length, :real_length]
    # Renormalize across keys (real positions only — they already are if SDPA
    # masked properly, but be defensive after slicing).
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    # Fraction of attention onto the prompt span — sum over key dim.
    prompt_mass = attn[..., :prompt_token_count].sum(dim=-1)              # (n_layers, n_heads, n_resp)
    return prompt_mass.mean(dim=-1).flatten()                              # (n_layers * n_heads,)


def extract_logit_lens_features(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_token_count: int,
    lm_head_weight: torch.Tensor,
    final_norm: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Project intermediate residual states through the LM head and measure how
    the model's next-token predictions evolve across layers.

    Truthful responses should crystallize early in the residual stream and
    persist; hallucinated responses should "flip" late in the network as
    the late layers fabricate something the early layers didn't predict.

    Args:
        hidden_states:       (n_layers, seq_len, hidden_dim) for one sample.
                             Layer 0 is the embedding; last index is the
                             final transformer output.
        input_ids:           (seq_len,) integer token ids.
        attention_mask:      (seq_len,) 1 for real tokens, 0 for padding.
        prompt_token_count:  number of tokens in the prompt portion.
        lm_head_weight:      (vocab_size, hidden_dim) tied/untied LM head matrix.
        final_norm:          RMSNorm module applied before LM head in the real
                             forward pass. If provided, we apply it to each
                             layer's hidden state before projecting (proper
                             "tuned-lens-like" path). If None, naive logit lens.

    Returns:
        1-D tensor of 10 features.
    """
    last_pos = int(attention_mask.nonzero(as_tuple=False)[-1].item())
    real_length = last_pos + 1

    if prompt_token_count >= real_length - 1:
        return torch.zeros(10, device=hidden_states.device)

    # Response token positions whose prediction we score:
    # logits at position t-1 predict token at position t. The last *predictable*
    # response token is at position last_pos, predicted by logits at last_pos-1.
    pred_positions = list(range(max(prompt_token_count - 1, 0), real_length - 1))
    target_positions = [p + 1 for p in pred_positions]
    if len(pred_positions) == 0:
        return torch.zeros(10, device=hidden_states.device)

    target_ids = input_ids[target_positions]                    # (n_resp,)
    n_resp = len(pred_positions)
    n_layers = hidden_states.size(0)

    # Subsample the layers we project — 25 full-vocab softmaxes is too much
    # memory on this host. 7 scan layers covers the full depth at low cost.
    SCAN_LAYERS = [0, 4, 8, 12, 16, 20, n_layers - 1]
    KL_LAYERS = {8, 12, 16, 20}

    top1_dict = {}                                              # layer -> (n_resp,) top-1 ids
    nll_dict = {}                                               # layer -> (n_resp,) NLL at target
    final_log_probs = None
    kl_to_final = {}

    for l in SCAN_LAYERS:
        h_l = hidden_states[l, pred_positions, :]               # (n_resp, hidden_dim)
        if final_norm is not None:
            h_l = final_norm(h_l)
        logits_l = h_l @ lm_head_weight.T                       # (n_resp, vocab)
        top1_dict[l] = logits_l.argmax(dim=-1)
        # NLL at target via logits[target] - logsumexp(logits). Cheaper than
        # full log_softmax tensor, but still O(vocab) — unavoidable.
        target_logits = logits_l.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
        lse = torch.logsumexp(logits_l.float(), dim=-1)
        nll_dict[l] = -(target_logits.float() - lse)
        # Only store full log_probs when we actually need them (KL layers + final).
        if l == n_layers - 1 or l in KL_LAYERS:
            log_probs_l = torch.nn.functional.log_softmax(logits_l.float(), dim=-1)
            if l == n_layers - 1:
                final_log_probs = log_probs_l
            if l in KL_LAYERS:
                kl_to_final[l] = log_probs_l
        del logits_l

    final_probs = final_log_probs.exp()
    kl_features = []
    for L in (8, 12, 16, 20):
        if L not in kl_to_final:
            kl_features.append(torch.zeros((), device=hidden_states.device))
            continue
        layer_log_p = kl_to_final[L]
        kl_per_pos = (final_probs * (final_log_probs - layer_log_p)).sum(dim=-1)
        kl_features.append(kl_per_pos.mean())

    # "Layer at which top-1 first matches the final layer's top-1", per position.
    final_top1 = top1_dict[n_layers - 1]                          # (n_resp,)
    first_match_layer = torch.full(
        (n_resp,), float(n_layers), device=hidden_states.device
    )
    for l in SCAN_LAYERS:
        newly_matched = (top1_dict[l] == final_top1) & (first_match_layer == float(n_layers))
        first_match_layer = torch.where(
            newly_matched, torch.full_like(first_match_layer, float(l)), first_match_layer
        )

    late_commit_frac = (first_match_layer >= (n_layers - 4)).float().mean()

    # NLL trajectory features. Stack only the scan-layers' NLL.
    nll_stack = torch.stack([nll_dict[l] for l in SCAN_LAYERS], dim=0)  # (n_scan, n_resp)
    nll_var_across_layers = nll_stack.var(dim=0).mean()
    nll_drift_mid_to_final = (nll_dict[n_layers - 1] - nll_dict[12]).mean()

    features = torch.stack([
        first_match_layer.mean(),
        first_match_layer.max(),
        late_commit_frac,
        nll_var_across_layers,
        nll_drift_mid_to_final,
        *kl_features,
        nll_dict[n_layers - 1].mean(),
    ])
    return features.float()


def extract_perplexity_features(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_token_count: int,
) -> torch.Tensor:
    """Per-token NLL / probability statistics over the response span.

    Causal-LM logits at position ``t-1`` predict the token at position ``t``.
    We compute these statistics for the *response* tokens only (positions
    ``[prompt_token_count, real_length)``), so the features reflect how well
    Qwen models its own generated answer given the prompt context.

    Returns:
        1-D tensor of length 6:
        [mean_nll, max_nll, std_nll, mean_top1_prob, mean_rank, max_rank].
    """
    last_pos = int(attention_mask.nonzero(as_tuple=False)[-1].item())
    real_length = last_pos + 1

    # logits[t] predicts token at position t+1. Last usable logit position is
    # real_length - 2 (its target is the last real token at real_length - 1).
    if prompt_token_count >= real_length or real_length < 2:
        return torch.zeros(11, device=logits.device)

    pred_start = max(prompt_token_count - 1, 0)
    pred_end = real_length - 2
    if pred_end < pred_start:
        return torch.zeros(11, device=logits.device)
    pred_logits = logits[pred_start:pred_end + 1]                 # (n_resp, vocab)
    target_ids = input_ids[pred_start + 1:pred_end + 2]           # (n_resp,)

    log_probs = torch.nn.functional.log_softmax(pred_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    nll = -token_log_probs

    top1_prob = log_probs.max(dim=-1).values.exp()

    sorted_indices = log_probs.argsort(dim=-1, descending=True)
    rank_of_true = (sorted_indices == target_ids.unsqueeze(-1)).int().argmax(dim=-1).float()

    # ── trajectory shape features (where in the response does NLL spike?) ──
    n = nll.shape[0]
    if n >= 3:
        # Position of max NLL in [0, 1].
        pos_of_max = nll.argmax().float() / max(n - 1, 1)
        # Lag-1 autocorrelation of NLL across response tokens.
        if n >= 4 and nll.std() > 1e-6:
            nll_mean = nll.mean()
            x = nll[:-1] - nll_mean
            y = nll[1:] - nll_mean
            autocorr1 = (x * y).sum() / ((x * x).sum().clamp(min=1e-12).sqrt() *
                                         (y * y).sum().clamp(min=1e-12).sqrt())
        else:
            autocorr1 = torch.zeros((), device=nll.device)
        # Linear slope of NLL vs position (response progresses → does NLL drift?).
        positions = torch.arange(n, device=nll.device).float()
        pos_centered = positions - positions.mean()
        nll_centered = nll - nll.mean()
        slope = (pos_centered * nll_centered).sum() / (pos_centered * pos_centered).sum().clamp(min=1e-12)
        # Kurtosis-like spikiness measure (4th moment / variance^2 - 3).
        if nll.std() > 1e-6:
            zscored = (nll - nll.mean()) / nll.std().clamp(min=1e-6)
            kurt = (zscored ** 4).mean() - 3.0
        else:
            kurt = torch.zeros((), device=nll.device)
        # Prefix vs suffix NLL ratio (first third vs last third).
        third = max(1, n // 3)
        prefix_mean = nll[:third].mean()
        suffix_mean = nll[-third:].mean()
        prefix_suffix_ratio = prefix_mean / suffix_mean.clamp(min=1e-6)
    else:
        pos_of_max = torch.zeros((), device=nll.device)
        autocorr1 = torch.zeros((), device=nll.device)
        slope = torch.zeros((), device=nll.device)
        kurt = torch.zeros((), device=nll.device)
        prefix_suffix_ratio = torch.ones((), device=nll.device)

    return torch.stack([
        nll.mean(),
        nll.max(),
        nll.std() if nll.numel() > 1 else torch.zeros((), device=nll.device),
        top1_prob.mean(),
        rank_of_true.mean(),
        rank_of_true.max(),
        pos_of_max,
        autocorr1,
        slope,
        kurt,
        prefix_suffix_ratio,
    ])


_HEDGE_PHRASES = (
    "approximately", "i think", "i believe", "may have", "might be",
    "possibly", "perhaps", "around ", "about ", "i'm not sure", "unclear",
    "estimated", "roughly", "presumably", "i guess", "i suppose",
    "seems", "appears to", "likely", "probably",
)


def extract_heuristic_features(prompt_text: str, response_text: str) -> torch.Tensor:
    """Hand-crafted text-based features. Per-sample, no model required.

    Returns:
        1-D tensor of 9 features:
        [resp_chars, resp_words, resp_sents, hedge_count, prompt_overlap_frac,
         has_idk, has_user_prefix, has_assistant_prefix, response_starts_caps].
    """
    import re
    resp_lower = response_text.lower()
    prompt_lower = prompt_text.lower()

    resp_chars = len(response_text)
    resp_words = max(len(response_text.split()), 1)
    resp_sents = max(len(re.split(r"[.!?]+", response_text.strip())), 1)
    hedge_count = sum(1 for h in _HEDGE_PHRASES if h in resp_lower)
    prompt_words = set(prompt_lower.split())
    resp_word_list = resp_lower.split()
    overlap = sum(1 for w in resp_word_list if w in prompt_words) / max(len(resp_word_list), 1)
    has_idk = 1.0 if ("i don't know" in resp_lower or "i do not know" in resp_lower) else 0.0
    has_user_prefix = 1.0 if resp_lower.startswith("user:") else 0.0
    has_assistant_prefix = 1.0 if resp_lower.startswith("assistant:") else 0.0
    response_starts_caps = 1.0 if response_text and response_text[0].isupper() else 0.0

    return torch.tensor([
        float(resp_chars),
        float(resp_words),
        float(resp_sents),
        float(hedge_count),
        float(overlap),
        has_idk,
        has_user_prefix,
        has_assistant_prefix,
        response_starts_caps,
    ], dtype=torch.float32)


def compute_knn_features(
    X_train,
    X_query=None,
    k_list: tuple[int, ...] = (3, 5, 10),
    leave_one_out: bool = False,
):
    """KNN-OOD features — distances and cosine similarity to nearest neighbors.

    Hallucinated responses may live in less dense regions of the embedding
    space; their distance-to-k-th-nearest-training-neighbor differs from
    truthful responses.

    Args:
        X_train:        (N_train, D) ndarray — the neighbor pool.
        X_query:        (N_query, D) ndarray — points to compute features for.
                        If None, defaults to X_train and forces leave-one-out.
        k_list:         which k-NN distances to compute (e.g., 3rd, 5th, 10th NN).
        leave_one_out:  if True, treat each X_query sample as if absent from
                        X_train (used when X_query is a subset of X_train).

    Returns:
        np.ndarray of shape (N_query, len(k_list) + 2): per-k NN distances,
        mean of top-max(k) distances, cosine similarity to top-1 NN.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    if X_query is None:
        X_query = X_train
        leave_one_out = True

    max_k = max(k_list)
    extra_k = 1 if leave_one_out else 0

    nn = NearestNeighbors(n_neighbors=max_k + extra_k, algorithm="auto", metric="euclidean")
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_query)

    if leave_one_out:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    cols = []
    for k in k_list:
        cols.append(distances[:, k - 1])
    cols.append(distances.mean(axis=1))
    top1_idx = indices[:, 0]
    norms_q = np.linalg.norm(X_query, axis=1) + 1e-12
    norms_t = np.linalg.norm(X_train[top1_idx], axis=1) + 1e-12
    dot = (X_query * X_train[top1_idx]).sum(axis=1)
    cols.append(dot / (norms_q * norms_t))

    return np.column_stack(cols)


def compute_lid_features(X_train, X_query=None, leave_one_out: bool = False):
    """Two-NN local intrinsic dimensionality estimate (Facco et al. 2017).

    For each sample, compute the ratio mu = d2/d1 between its 2nd and 1st
    nearest neighbour distances. The intrinsic dimensionality is inversely
    related to how this ratio distributes; per-sample log(mu) is itself a
    useful "off-manifold-ness" feature.

    Returns:
        (N_query, 2) ndarray — [mu (ratio d2/d1), log(mu)] per sample.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    if X_query is None:
        X_query = X_train
        leave_one_out = True

    extra = 1 if leave_one_out else 0
    nn = NearestNeighbors(n_neighbors=2 + extra, algorithm="auto", metric="euclidean").fit(X_train)
    distances, _ = nn.kneighbors(X_query)
    if leave_one_out:
        distances = distances[:, 1:]
    d1 = distances[:, 0] + 1e-12
    d2 = distances[:, 1] + 1e-12
    mu = d2 / d1
    return np.column_stack([mu, np.log(mu + 1e-12)])


def compute_mahalanobis_features(X_train, y_train, X_query):
    """Per-class Mahalanobis distance features.

    Fits a multivariate Gaussian to each class on the training set; reports
    the distance from each query point to each class's Gaussian, plus their
    ratio and difference. **Note:** uses a shrunk covariance estimator to
    keep the inversion stable in high dim.

    Returns:
        (N_query, 4) ndarray — [d_truthful, d_hallucinated, ratio, diff].
    """
    import numpy as np
    from sklearn.covariance import LedoitWolf

    def class_mahal(class_label: int):
        Xc = X_train[y_train == class_label]
        mu = Xc.mean(axis=0)
        cov = LedoitWolf().fit(Xc).covariance_
        try:
            cov_inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov + 1e-3 * np.eye(cov.shape[0]))
        diff = X_query - mu
        return np.sqrt(np.maximum((diff @ cov_inv * diff).sum(axis=1), 0.0))

    d_truthful = class_mahal(0)
    d_hallucinated = class_mahal(1)
    ratio = d_truthful / (d_hallucinated + 1e-12)
    return np.column_stack([d_truthful, d_hallucinated, ratio, d_truthful - d_hallucinated])


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    attentions: torch.Tensor | None = None,
    use_geometric: bool = False,
    prompt_token_count: int | None = None,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    parts = [aggregate(hidden_states, attention_mask)]

    if use_geometric:
        parts.append(extract_geometric_features(hidden_states, attention_mask))

    if attentions is not None:
        parts.append(extract_attention_features(attentions, attention_mask, prompt_token_count))

    return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
