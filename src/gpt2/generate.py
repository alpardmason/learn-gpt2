"""
Autoregressive text generation for GPT-2.

Components:
  top_k_filter    — keep only the k highest-probability tokens (Task 13a)
  top_p_filter    — keep only the top-probability tokens summing to p (Task 13b)
  generate        — autoregressive sampling loop (Task 13c)

[theory] Autoregressive generation: at each step, the model computes a
probability distribution over the vocabulary given all previous tokens.
We sample the next token from this distribution, append it, and repeat.

Raw sampling from the full distribution often produces incoherent text because
low-probability tokens can be selected. Two widely-used filtering strategies:

  Top-k: keep only the k tokens with the highest logits; zero out the rest.
         This eliminates the long tail of very unlikely tokens.

  Top-p (nucleus sampling, Holtzman et al. 2020): sort tokens by probability
         descending; keep the smallest set whose cumulative probability ≥ p.
         This is adaptive: the nucleus size grows when the distribution is
         flat (uncertain) and shrinks when the model is confident.

Temperature scales the logits before softmax:
  logits_scaled = logits / temperature
  temperature > 1 → flatter distribution (more random)
  temperature < 1 → sharper distribution (more greedy)
  temperature = 0 → argmax (greedy decoding, deterministic)

[modern] Production systems use KV-caching: instead of recomputing the K and V
tensors for all past tokens on every step, they are cached and reused. This
reduces the per-step compute from O(T²) to O(T). KV-cache is NOT implemented
here to keep the code simple.

[modern] Speculative decoding (Chen et al. 2023) uses a small draft model to
propose multiple tokens at once, then verifies them with the large model in a
single forward pass. This achieves 2-3× throughput on GPU.
"""

from __future__ import annotations

import torch

from gpt2.model import GPT2Model

# ===========================================================================
# Task 13a — Top-k Filtering
# ===========================================================================

# Warmup -----------------------------------------------------------------------
# Given logits = [1.0, 2.0, 0.5, 3.0] and k=2:
#   Step 1: sort descending → [3.0, 2.0, 1.0, 0.5]
#   Step 2: the k-th largest (k=2) value is 2.0
#   Step 3: zero out all logits < 2.0 → [?, 2.0, ?, 3.0]
#   What values replace the zeroed-out positions? (Hint: −∞)
#   Which two tokens survive the filter? (Hint: indices 1 and 3)
# ------------------------------------------------------------------------------


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Zero out all logits except the k largest.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (..., vocab_size). Typically (1, vocab_size) for generation.
    k : int
        Number of top logits to keep. Must be ≥ 1.

    Returns
    -------
    torch.Tensor
        Same shape as input, with all but the top-k values replaced by −∞.
        (Setting to −∞ ensures those tokens get probability ≈ 0 after softmax.)

    Steps
    -----
    1. Use torch.topk to find the k-th largest value along dim=-1.
    2. Create a mask: positions where logits < threshold → True.
    3. Use masked_fill to set those positions to -inf.
    """
    raise NotImplementedError(
        "Task 13a: Implement top_k_filter.\n"
        "  values, _ = torch.topk(logits, k, dim=-1)\n"
        "  threshold  = values[..., -1:]\n"
        "  return logits.masked_fill(logits < threshold, float('-inf'))"
    )


# ===========================================================================
# Task 13b — Top-p (Nucleus) Filtering
# ===========================================================================

# Warmup -----------------------------------------------------------------------
# Given probabilities (after softmax) = [0.5, 0.3, 0.15, 0.04, 0.01] and p=0.9:
#   Step 1: sort descending → [0.5, 0.3, 0.15, 0.04, 0.01]
#   Step 2: cumulative sum  → [0.5, 0.8, 0.95, 0.99, 1.00]
#   Step 3: which is the first index where cum_sum ≥ 0.9? (index 2, value 0.95)
#   Step 4: keep indices 0..2; zero out the rest.
#
# Note: we do the filtering on LOGITS (before softmax), not on probabilities,
# to avoid computing softmax twice. The logic is the same because sorting by
# logit value is equivalent to sorting by probability.
# ------------------------------------------------------------------------------


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Zero out logits outside the top-p nucleus.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (..., vocab_size).
    p : float
        Cumulative probability threshold in (0, 1]. p=1.0 keeps everything.

    Returns
    -------
    torch.Tensor
        Same shape as input, with tokens outside the nucleus set to −∞.

    Steps
    -----
    1. Sort logits descending; keep track of original indices.
    2. Compute cumulative softmax probabilities of the sorted logits.
    3. Remove tokens where the cumulative probability exceeds p.
       (Shift by 1: include the token that pushes past p so sum is ≥ p.)
    4. Scatter the mask back to the original (unsorted) order.
    5. Apply mask: set removed tokens to -inf.
    """
    raise NotImplementedError(
        "Task 13b: Implement top_p_filter.\n"
        "  sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)\n"
        "  sorted_probs = F.softmax(sorted_logits, dim=-1)\n"
        "  cum_probs    = torch.cumsum(sorted_probs, dim=-1)\n"
        "  # Shift right so the token that pushes over p is included\n"
        "  remove = (cum_probs - sorted_probs) > p\n"
        "  sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))\n"
        "  # Unsort: scatter back to original token order\n"
        "  return sorted_logits.scatter(-1, sorted_idx, sorted_logits)"
    )


# ===========================================================================
# Task 13c — Autoregressive Generation Loop
# ===========================================================================

# Warmup -----------------------------------------------------------------------
# Trace ONE step of the generation loop (B=1, T=3, vocab_size=64, n_embd=32):
#   context = [5, 12, 7]             shape of idx: (1, 3)
#   logits = model(idx)              shape: ?
#   logits_last = logits[:, -1, :]   shape: ?  (why only the last position?)
#   next_token = sample(logits_last) shape: ?
#
# Answer: (1,3,64) → (1,64) → (1,1)  then appended to idx → (1,4)
# ------------------------------------------------------------------------------


@torch.inference_mode()
def generate(
    model: GPT2Model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Autoregressively generate tokens from a prompt.

    Parameters
    ----------
    model : GPT2Model
        The (trained) GPT-2 model. Must be in eval mode.
    idx : torch.Tensor
        Prompt token IDs, shape (B, T). Values in [0, vocab_size).
    max_new_tokens : int
        Number of new tokens to generate.
    temperature : float
        Softmax temperature. 1.0 = unchanged; < 1 = sharper; > 1 = flatter.
        Set to 0.0 for greedy (argmax) decoding.
    top_k : int | None
        If set, apply top-k filtering before sampling.
    top_p : float | None
        If set, apply nucleus (top-p) filtering before sampling.
    seed : int | None
        Optional random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Token IDs, shape (B, T + max_new_tokens), including the original prompt.

    Steps (implement in order)
    --------------------------
    For each step in range(max_new_tokens):
      1. Crop context to last n_ctx tokens (model's maximum context window).
      2. Forward pass: logits = model(idx_crop).
      3. Extract last-position logits: logits = logits[:, -1, :].
      4. If temperature == 0.0: return argmax (greedy).
      5. Scale logits by temperature: logits /= temperature.
      6. Apply top_k_filter if top_k is set.
      7. Apply top_p_filter if top_p is set.
      8. Sample: probs = softmax(logits); next_id = multinomial(probs, 1).
      9. Append next_id to idx and continue.
    """
    raise NotImplementedError(
        "Task 13c: Implement the generate function.\n"
        "  Key calls:\n"
        "    idx_crop = idx[:, -model.config.n_ctx:]\n"
        "    next_id  = torch.multinomial(probs, num_samples=1)\n"
        "    idx      = torch.cat([idx, next_id], dim=1)\n"
        "  Remember: @torch.inference_mode() is already applied by the decorator."
    )
