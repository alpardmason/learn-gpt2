"""
GPT-2 model definition in PyTorch.

Component hierarchy (bottom-up):
  GELU                    — activation function
  LayerNorm               — normalisation layer
  CausalSelfAttention     — multi-head causal attention
  MLP                     — position-wise feed-forward network
  Block                   — one transformer block (attn + mlp + residuals)
  GPT2Model               — full model (embeddings + N blocks + head)

[theory] GPT-2 is a decoder-only transformer (Vaswani et al., 2017) with two key
differences from the original "Attention is All You Need":
  1. Pre-LayerNorm: normalisation is applied BEFORE each sub-layer (not after).
     This improves gradient flow in deep networks.
  2. No encoder: there is no cross-attention, only causal self-attention.

[modern] Modern LLMs (LLaMA, Mistral, Gemma) diverge from GPT-2 in several ways:
  • RMSNorm instead of LayerNorm (cheaper, no mean subtraction)
  • RoPE instead of learned absolute position embeddings
  • SwiGLU instead of GELU in the MLP
  • Grouped-Query Attention (GQA) instead of full multi-head attention
  • Untied lm_head and wte weights (GPT-2 ties them)

References:
  Paper:     references/language_models_are_unsupervised_multitask_learners.pdf
  OpenAI TF: references/gpt-2/src/model.py
  nanochat:  references/nanochat/nanochat/gpt.py  (PyTorch patterns only)
"""

import math

import torch
import torch.nn as nn

from gpt2.config import GPT2Config

# ===========================================================================
# Task 4 — GELU Activation
# ===========================================================================

# [theory] GELU (Gaussian Error Linear Unit, Hendrycks & Gimpel 2016) is defined as:
#
#   GELU(x) = x · Φ(x)
#
# where Φ(x) is the standard Gaussian CDF. Because Φ is expensive to compute,
# GPT-2 uses the tanh approximation (also used in the GELU paper):
#
#   GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
#
# Intuition: unlike ReLU which hard-gates at zero, GELU smoothly scales the input
# by the probability that it is positive. This soft gating helps gradients flow
# through the network.
#
# [modern] LLaMA 2/3 uses SwiGLU: SiLU(xW₁) ⊙ (xW₂), which has a third weight
# matrix and no approximation. GPT-4 is rumoured to also use a gated activation.
#
# Reference: references/gelu_paper.pdf

# Warmup -----------------------------------------------------------------------
# Compute the following by hand using the tanh approximation formula:
#   GELU(0.0)  = ?   (hint: tanh(0) = 0, so the answer is simple)
#   GELU(1.0)  = ?   (hint: √(2/π) ≈ 0.7979, 0.044715 · 1³ ≈ 0.045)
# Check: GELU(0.0) should be exactly 0.0.
#        GELU(1.0) should be approximately 0.8413.
# ------------------------------------------------------------------------------


_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


class GELU(nn.Module):
    """
    Tanh-approximation GELU activation, as used in GPT-2.

    This is equivalent to torch.nn.GELU(approximate='tanh').
    We implement it from scratch to understand the formula.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU elementwise.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            Same shape as input, with GELU applied elementwise.

        Formula
        -------
        0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        """
        # Original:
        # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * (x**3))))
        return 0.5 * x * (1 + torch.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x.pow(3))))


# ===========================================================================
# Task 5 — Layer Normalisation
# ===========================================================================

# [theory] Layer Normalisation (Ba et al., 2016) normalises across the feature
# dimension (last axis) of each token independently:
#
#   μ = mean(x)        (scalar per token)
#   σ² = var(x)        (scalar per token)
#   x̂ = (x − μ) / √(σ² + ε)
#   out = γ · x̂ + β   (γ, β are learnable per-feature parameters)
#
# Compared to BatchNorm, LayerNorm does NOT depend on batch statistics, making it
# more stable for variable-length sequences and small batches.
#
# ε (epsilon) prevents division by zero when σ² ≈ 0. GPT-2 uses ε = 1e-5.
#
# [modern] RMSNorm (Zhang & Sennrich, 2019) drops the mean subtraction:
#   x̂ = x / RMS(x)  where RMS(x) = √(mean(x²))
# This is ~10% cheaper and works just as well in practice (LLaMA, Mistral).
#
# Reference: references/gpt-2/src/model.py  norm(x, scope, ...)

# Warmup -----------------------------------------------------------------------
# Normalize the vector [1.0, 2.0, 3.0] by hand:
#   Step 1: μ = ?
#   Step 2: σ² = ?
#   Step 3: x̂ = ?
# Assume ε = 0 and γ = [1,1,1], β = [0,0,0] (no learnable transform).
# Your answer: x̂ = [-1.2247, 0.0, 1.2247] (approximately).
# ------------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """
    Layer normalisation with learnable affine transform.

    Parameters
    ----------
    n_embd : int
        Size of the last dimension (feature dimension) to normalise over.
    eps : float
        Small constant added to variance for numerical stability.
    """

    def __init__(self, n_embd: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # [best practice] Use nn.Parameter so PyTorch tracks these for
        # autograd and includes them in model.parameters().
        self.weight = nn.Parameter(torch.ones(n_embd))  # γ (scale), init 1
        self.bias = nn.Parameter(torch.zeros(n_embd))  # β (shift), init 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise x along its last dimension, then apply affine transform.

        Parameters
        ----------
        x : torch.Tensor
            Shape (..., n_embd). Typically (B, T, C) in transformer context.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        # [best practice] Population variance via x.var(unbiased=False).
        #   Pro: Matches the LayerNorm paper and nn.LayerNorm; one readable line.
        #   Con: x.var() recomputes mean internally — two reduction passes over x.
        #   Alternative: var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        #     Pro: reuses the mean below; one fewer full reduction.
        #     Con: slightly less obvious that this is population variance.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # [best practice] sqrt (current) vs rsqrt — mathematically equivalent:
        #   (x - mean) / (var + eps).sqrt()   divide by σ — readable, matches the paper
        #   (x - mean) * (var + eps).rsqrt()  multiply by 1/σ — faster on GPU (mul < div)
        x_hat = (x - mean) / (var + self.eps).sqrt()

        # [best practice] Production alternative — F.layer_norm(x, (n_embd,), weight, bias, eps)
        #   Pro: fused CUDA kernel; fastest at training/inference scale.
        #   Con: hides the normalization math; keep the explicit form for learning.
        return self.weight * x_hat + self.bias


# ===========================================================================
# Task 6 — Causal Multi-Head Self-Attention
# ===========================================================================

# [theory] Scaled Dot-Product Attention (Vaswani et al., 2017):
#
#   Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) · V
#
# where d_k = head_dim = n_embd / n_head. The 1/√d_k scale prevents the dot
# products from becoming very large (which would push softmax into saturation).
#
# Multi-Head Attention runs h independent attention functions in parallel:
#   head_i = Attention(Q Wᵢ_Q, K Wᵢ_K, V Wᵢ_V)
#   MultiHead(Q,K,V) = concat(head_1, ..., head_h) W_O
#
# Causal (autoregressive) masking ensures token i can only attend to tokens
# j ≤ i. This is enforced by adding −∞ to positions j > i before softmax,
# making those attention weights exactly zero.
#
# [modern] Modern improvements:
#   • RoPE: positional information is encoded in Q and K via rotation, not
#     in the input embeddings. This generalises better to longer sequences.
#   • GQA/MQA (Grouped-Query / Multi-Query Attention): K and V have fewer
#     heads than Q, reducing KV-cache memory during inference.
#   • FlashAttention: fused kernel that avoids materialising the full (T×T)
#     attention matrix in HBM, enabling longer contexts.
#
# Reference: references/gpt-2/src/model.py  attn(x, scope, n_state, ...)
# [nanochat] We use F.scaled_dot_product_attention (PyTorch ≥ 2.0) which
#            dispatches to FlashAttention when available. The original OpenAI
#            code implements the attention kernel manually in TensorFlow.

# Warmup -----------------------------------------------------------------------
# For a sequence of length T=4, draw the causal attention mask.
# Which cells are 1 (can attend) and which are 0 (masked)?
#
#        pos 0  pos 1  pos 2  pos 3
# pos 0 [  ?     ?     ?     ?  ]
# pos 1 [  ?     ?     ?     ?  ]
# pos 2 [  ?     ?     ?     ?  ]
# pos 3 [  ?     ?     ?     ?  ]
#
# Answer: it is a lower-triangular matrix of 1s. The mask is "causal" because
# token at position i can see the past (j < i) and itself (j = i) but NOT the
# future (j > i).
# ------------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    forward() implements scaled dot-product attention manually so the math is
    visible (QK^T / sqrt(d_k) → causal mask → softmax → AV). For production
    training, replace the manual kernel with F.scaled_dot_product_attention
    (see inline comments in forward).

    [best practice] PyTorch SDPA with is_causal=True auto-selects
    FlashAttention, memory-efficient attention, or math attention depending on
    hardware and input shape — without materialising the full (T×T) score matrix.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # [theory] A single linear layer projects x to Q, K, V concatenated.
        # This is the same as three separate projections but one matrix multiply
        # is faster in practice. Output dim is 3 * n_embd.
        # Reference: references/gpt-2/src/model.py  conv1d(x, 'c_attn', n_state*3)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection: maps the concatenated head outputs back to n_embd.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # [best practice] Scale residual projections by 1/√(2·n_layer) as in
        # the GPT-2 paper §2: "We scale the weights of residual layers at
        # initialization by a factor of 1/√N where N is the number of residual
        # layers." We store n_layer on the module so _init_weights can use it.
        self._n_layer = config.n_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head causal self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, C) where B=batch, T=sequence length, C=n_embd.

        Returns
        -------
        torch.Tensor
            Shape (B, T, C) — the attended output.

        Steps (implement these in order):
        1. Project x → Q, K, V  via self.c_attn  (shape: B, T, 3C)
        2. Split along last dim: q, k, v each (B, T, C)
        3. Reshape to separate heads: (B, n_head, T, head_dim)
           Hint: x.view(B, T, n_head, head_dim).transpose(1, 2)
        4. Scaled dot-product attention (manual below; SDPA one-liner in comments)
        5. Merge heads back: (B, T, C)
           Hint: .transpose(1, 2).contiguous().view(B, T, C)
        6. Apply output projection self.c_proj
        """
        batch_size = x.size(0)
        qkv: torch.Tensor = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=-1)  # 3 * (B, T, C)

        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)

        # --- Manual attention (pedagogical) -----------------------------------
        # [best practice] For production, replace this entire block with:
        #   import torch.nn.functional as F
        #   output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # SDPA fuses scale → causal mask → softmax → weighted sum and can
        # dispatch to FlashAttention, avoiding O(B·H·T²) memory for score tensors.
        # At GPT-2 Small (T=1024, H=12), one (B,H,T,T) float32 tensor ≈ 48 MB;
        # manual attention allocates several per layer per forward pass.

        # Attention scores: QK^T / sqrt(head_dim)  →  shape (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Causal mask: token i may attend only to j ≤ i.
        # [best practice] SDPA applies this implicitly via is_causal=True — no
        # tril(ones_like) allocation each forward pass.
        mask = torch.tril(torch.ones_like(scores)).bool()
        scores = torch.where(mask, scores, float("-inf"))

        # Row-wise softmax → attention weights (each row sums to 1)
        p_attn = torch.softmax(scores, dim=-1)

        # Weighted sum of values:  output = P @ V
        output = torch.matmul(p_attn, v)
        # --- End manual attention ---------------------------------------------

        # Merge heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_embd)

        # Apply output projection
        return self.c_proj(output)


# ===========================================================================
# Task 7 — MLP (Feed-Forward Network)
# ===========================================================================

# [theory] The position-wise feed-forward network in GPT-2 is a 2-layer MLP:
#
#   FFN(x) = GELU(x W₁ + b₁) W₂ + b₂
#
# The inner dimension is 4 × n_embd. This "4×" expansion is a design choice
# from the original Transformer paper with no rigorous theoretical justification,
# but it has proven effective empirically. The FFN adds non-linearity and
# capacity while attention handles position mixing.
#
# [modern] LLaMA / Mistral / Gemma use SwiGLU (Shazeer 2020):
#   FFN_SwiGLU(x) = (SiLU(x W₁) ⊙ (x W₃)) W₂
# This requires THREE weight matrices instead of two, and the expansion factor
# is typically ~8/3 × n_embd (not 4×) to keep parameter counts comparable.
#
# Reference: references/gpt-2/src/model.py  mlp(x, scope, n_state, ...)

# Warmup -----------------------------------------------------------------------
# With n_embd = 32, what are the weight shapes of c_fc and c_proj?
#   c_fc.weight  shape: ?
#   c_proj.weight shape: ?
# Answer: c_fc.weight is (128, 32), c_proj.weight is (32, 128).
# (PyTorch linear layer stores weights as (out_features, in_features).)
# ------------------------------------------------------------------------------


class MLP(nn.Module):
    """
    Position-wise feed-forward network (2-layer MLP with GELU).

    Inner dimension is 4 × n_embd, matching the original GPT-2 paper.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply two-layer MLP with GELU activation.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, C).

        Returns
        -------
        torch.Tensor
            Shape (B, T, C). (Dimension is preserved end-to-end.)
        """
        return self.c_proj(self.gelu(self.c_fc(x)))


# ===========================================================================
# Task 8 — Transformer Block
# ===========================================================================

# [theory] Each GPT-2 transformer block consists of:
#
#   x = x + Attention(LayerNorm₁(x))   ← self-attention with residual
#   x = x + MLP(LayerNorm₂(x))         ← feed-forward with residual
#
# This is the PRE-NORM variant: LayerNorm is applied BEFORE each sub-layer.
# The original "Attention is All You Need" uses POST-NORM (LayerNorm after the
# residual add). Pre-norm improves gradient flow and training stability in deep
# networks, which is why GPT-2 and virtually all subsequent LLMs use it.
#
# The residual connections (x = x + ...) serve two purposes:
#   1. Gradient highway: gradients flow directly through the residual path,
#      mitigating the vanishing gradient problem.
#   2. Identity initialisation: at initialisation, the residual branches are
#      near zero so the network starts as a near-identity function, making early
#      training stable.
#
# [modern] The architecture is unchanged in most modern LLMs; the main variation
# is what goes inside the attention and MLP sub-layers (RoPE, GQA, SwiGLU, etc.).
#
# Reference: references/gpt-2/src/model.py  block(x, scope, ...)

# Warmup -----------------------------------------------------------------------
# Trace the residual stream for a single block on input x (shape B,T,C):
#
#   a = LayerNorm₁(x)          shape: ?
#   b = Attention(a)            shape: ?
#   x = x + b                  shape: ?  ← first residual add
#   c = LayerNorm₂(x)          shape: ?
#   d = MLP(c)                  shape: ?
#   x = x + d                  shape: ?  ← second residual add
#
# All intermediate shapes should be (B, T, C). The block is shape-preserving.
# ------------------------------------------------------------------------------


class Block(nn.Module):
    """
    One GPT-2 transformer block: pre-norm attention + pre-norm MLP.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one transformer block with pre-norm residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, C) — the residual stream.

        Returns
        -------
        torch.Tensor
            Shape (B, T, C).
        """
        raise NotImplementedError(
            "Task 8: Implement Block.forward.\n"
            "  x = x + self.attn(self.ln_1(x))\n"
            "  x = x + self.mlp(self.ln_2(x))\n"
            "  Reference: references/gpt-2/src/model.py  block(x, scope, ...)"
        )


# ===========================================================================
# Task 9 — GPT-2 Model (Full Forward Pass)
# ===========================================================================

# [theory] The full GPT-2 forward pass:
#
#   1. Token embedding:    h = wte[token_ids]           shape: (B, T, C)
#   2. Position embedding: h = h + wpe[0:T]             shape: (B, T, C)
#   3. N transformer blocks: h = Block_i(h)  for i in 0..N-1
#   4. Final LayerNorm:    h = ln_f(h)
#   5. Logits (weight-tied): logits = h @ wte.weightᵀ  shape: (B, T, vocab_size)
#
# WEIGHT TYING: the token embedding matrix wte.weight is REUSED as the output
# projection (lm_head). This was proposed by Press & Wolf (2017). Intuition:
# the embedding encodes each token as a direction in d_model-space; the output
# projection should use the same directions to score how likely each token is.
# Weight tying also halves the number of parameters in the embedding layer.
#
# Position embeddings: GPT-2 uses LEARNED absolute position embeddings (wpe).
# Each position 0..n_ctx-1 has its own embedding vector trained from scratch.
#
# [modern] Learned absolute positions have been replaced by:
#   • RoPE (Rotary Position Embedding) in LLaMA, Mistral, GPT-NeoX: encodes
#     relative positions via rotation of Q and K vectors; generalises to longer
#     sequences than seen during training (with NTK-aware scaling).
#   • ALiBi (Attention with Linear Biases) in BLOOM: adds a linear bias to
#     attention scores based on distance.
#
# Reference: references/gpt-2/src/model.py  model(hparams, X, ...)
# [nanochat] Module structure (nn.ModuleDict) adapted from nanochat/gpt.py

# Warmup -----------------------------------------------------------------------
# With the tiny test config (vocab_size=64, n_ctx=16, n_embd=32, n_head=4, n_layer=2):
#   Input  ids: shape (2, 8)   (B=2, T=8)
#
#   After wte:  shape ?
#   After wpe:  shape ?  (what is added to wte output?)
#   After blocks: shape ?
#   After ln_f: shape ?
#   logits: shape ?     (remember: weight-tied projection to vocab_size=64)
#
# Answer: (2,8,32) → (2,8,32) → (2,8,32) → (2,8,32) → (2,8,64)
# ------------------------------------------------------------------------------


class GPT2Model(nn.Module):
    """
    Full GPT-2 language model.

    Attributes
    ----------
    transformer.wte : nn.Embedding
        Token embedding table, shape (vocab_size, n_embd).
    transformer.wpe : nn.Embedding
        Position embedding table, shape (n_ctx, n_embd).
    transformer.h : nn.ModuleList
        List of N transformer Blocks.
    transformer.ln_f : LayerNorm
        Final layer norm applied after all blocks.

    Note
    ----
    The language model head (logit projection) is NOT a separate parameter.
    It reuses transformer.wte.weight via weight tying.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config

        # [nanochat] nn.ModuleDict groups sub-modules under a namespace, making
        # state_dict keys look like "transformer.wte.weight" — matching the
        # naming convention of the official GPT-2 release and HuggingFace.
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.n_ctx, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd),
            }
        )

        # Weight tying: lm_head shares weights with token embedding.
        # We do NOT register lm_head as nn.Linear to avoid duplicate parameters.
        # Instead, we call F.linear with the embedding weight in forward().
        # [best practice] Always tie weights explicitly and document it; forgetting
        # weight tying is a common source of parameter count discrepancies.

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise model weights as described in the GPT-2 paper §2:
          • Embedding weights ~ N(0, 0.02)
          • Linear weights    ~ N(0, 0.02)
          • Residual projections scaled by 1/√(2 · n_layer)

        [theory] The residual scaling prevents the residual stream from growing too
        large as depth increases. With N blocks, the signal accumulates N times, so
        each residual branch is scaled down by √N to keep the total variance ~O(1).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual projection weights: c_proj in both attention and MLP
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return logits.

        Parameters
        ----------
        idx : torch.Tensor
            Integer token IDs, shape (B, T). Values must be in [0, vocab_size).
            T must be ≤ n_ctx.

        Returns
        -------
        torch.Tensor
            Logits of shape (B, T, vocab_size). Entry [b, t, v] is the
            unnormalised log-probability that the token at position t+1 in
            sequence b is token v.

        Steps (implement in order):
        1. Check T ≤ n_ctx.
        2. Build position indices: pos = torch.arange(0, T)  on same device as idx
        3. Embed: tok_emb = wte(idx);  pos_emb = wpe(pos)
        4. h = tok_emb + pos_emb
        5. For each block in transformer.h: h = block(h)
        6. h = transformer.ln_f(h)
        7. logits = h @ transformer.wte.weight.T   (weight-tied projection)
        """
        raise NotImplementedError(
            "Task 9: Implement GPT2Model.forward.\n"
            "  Remember: weight-tied logits = h @ self.transformer.wte.weight.T\n"
            "  Reference: references/gpt-2/src/model.py  model(hparams, X, ...)"
        )

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """
        Count trainable parameters.

        Parameters
        ----------
        exclude_embeddings : bool
            If True, exclude wte and wpe from the count. This matches the
            convention used in the GPT-2 paper when reporting model size.
        """
        params = list(self.parameters())
        if exclude_embeddings:
            params = [p for n, p in self.named_parameters() if "wte" not in n and "wpe" not in n]
        return sum(p.numel() for p in params)
