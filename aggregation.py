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
) -> torch.Tensor:
    """Attention-based features for hallucination detection.

    Args:
        attentions:     Tensor of shape ``(n_layers, n_heads, seq_len, seq_len)``
                        for one sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        1-D tensor of length ``3 * n_layers`` (per-layer entropy, self-attn weight,
        and top-3 concentration of the last-token attention distribution).
    """
    last_pos = int(attention_mask.nonzero(as_tuple=False)[-1].item())

    # Last-token attention distribution per layer, mean over heads.
    # Shape: (n_layers, seq_len).
    attn_from_last = attentions[:, :, last_pos, :].mean(dim=1)

    # Restrict to real tokens and renormalize so distributions sum to 1.
    real_mask = attention_mask.float().to(attn_from_last.device)
    attn_masked = attn_from_last * real_mask.unsqueeze(0)
    attn_norm = attn_masked / attn_masked.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    log_p = torch.log(attn_norm.clamp(min=1e-9))
    entropy = -(attn_norm * log_p).sum(dim=-1)                            # (n_layers,)
    self_attn = attn_norm[:, last_pos]                                    # (n_layers,)
    top3 = attn_norm.topk(min(3, attn_norm.size(-1)), dim=-1).values.sum(dim=-1)  # (n_layers,)

    return torch.cat([entropy, self_attn, top3], dim=0)


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
        return torch.zeros(6, device=logits.device)

    pred_start = max(prompt_token_count - 1, 0)
    pred_end = real_length - 2
    if pred_end < pred_start:
        return torch.zeros(6, device=logits.device)
    pred_logits = logits[pred_start:pred_end + 1]                 # (n_resp, vocab)
    target_ids = input_ids[pred_start + 1:pred_end + 2]           # (n_resp,)

    log_probs = torch.nn.functional.log_softmax(pred_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    nll = -token_log_probs

    top1_prob = log_probs.max(dim=-1).values.exp()

    sorted_indices = log_probs.argsort(dim=-1, descending=True)
    rank_of_true = (sorted_indices == target_ids.unsqueeze(-1)).int().argmax(dim=-1).float()

    return torch.stack([
        nll.mean(),
        nll.max(),
        nll.std() if nll.numel() > 1 else torch.zeros((), device=nll.device),
        top1_prob.mean(),
        rank_of_true.mean(),
        rank_of_true.max(),
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


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    attentions: torch.Tensor | None = None,
    use_geometric: bool = False,
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
        parts.append(extract_attention_features(attentions, attention_mask))

    return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
