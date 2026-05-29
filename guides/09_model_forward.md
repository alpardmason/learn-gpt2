# Guide 09 — GPT-2 Model (Full Forward Pass)

## Theory

The full GPT-2 forward pass has five stages:

**1. Token embedding:**
$$e_{\text{tok}} = \text{wte}[\text{idx}] \in \mathbb{R}^{B \times T \times C}$$

`wte` (word token embedding) is a lookup table of size `(vocab_size, n_embd)`. Each token ID maps to a learnable vector.

**2. Position embedding:**
$$e_{\text{pos}} = \text{wpe}[0:T] \in \mathbb{R}^{T \times C}$$

`wpe` (word position embedding) is a lookup table of size `(n_ctx, n_embd)`. Each position 0..T-1 maps to a learnable vector. GPT-2 uses **learned** positions (not sinusoidal).

**3. Input representation:**
$$h = e_{\text{tok}} + e_{\text{pos}}$$

**4. Transformer blocks:**
$$h = \text{Block}_N(\ldots \text{Block}_1(h))$$

**5. Final norm + logits (weight-tied):**
$$h = \text{ln\_f}(h)$$
$$\text{logits} = h \, W_{\text{te}}^\top \in \mathbb{R}^{B \times T \times V}$$

where $W_{\text{te}} \in \mathbb{R}^{V \times C}$ is the **same** weight matrix as the token embedding table.

**Weight tying** (Press & Wolf 2017): the output projection matrix is the
*same tensor* as the token embedding matrix `wte.weight`. This halves the
embedding parameter count (~38M for GPT-2 Small) and often improves
perplexity by encouraging consistent token representations throughout the model.

**Module layout:** sub-modules live under `self.transformer`, an `nn.ModuleDict`, so
`state_dict` keys match HuggingFace / OpenAI naming (`transformer.wte.weight`, etc.).
Access with bracket notation: `self.transformer["wte"]`, not attribute access.

### Matrix-form backpropagation

Let upstream loss gradient be $\frac{\partial \mathcal{L}}{\partial \text{logits}} \in \mathbb{R}^{B \times T \times V}$.

**Tied logit projection.** With $\text{logits}_{b,t,:} = h_{b,t,:} \, W_{\text{te}}^\top$ (matrix multiply $H W^\top$):

$$\frac{\partial \mathcal{L}}{\partial H} = \frac{\partial \mathcal{L}}{\partial \text{logits}} \, W_{\text{te}} \in \mathbb{R}^{B \times T \times C}$$

$$\frac{\partial \mathcal{L}}{\partial W_{\text{te}}} = \sum_{b,t} \left(\frac{\partial \mathcal{L}}{\partial \text{logits}}\right)_{b,t,:}^\top h_{b,t,:} \in \mathbb{R}^{V \times C}$$

Because $W_{\text{te}}$ is shared, gradients from the **embedding lookup** (stage 1) and the **output projection** (stage 5) **accumulate** on the same parameter tensor. That is why you must not register a separate `lm_head`.

**Input sum.** For $h = e_{\text{tok}} + e_{\text{pos}}$:

$$\frac{\partial \mathcal{L}}{\partial e_{\text{tok}}} = \frac{\partial \mathcal{L}}{\partial h}, \qquad \frac{\partial \mathcal{L}}{\partial e_{\text{pos}}} = \sum_b \frac{\partial \mathcal{L}}{\partial h_{b,:,:}}$$

Position embeddings aggregate gradients across the batch dimension.

**Block stack.** Each `Block` is shape-preserving $(B,T,C) \to (B,T,C)$; backprop chains through $N$ blocks via the standard residual-path Jacobians derived in guides 05–08.

**Jacobian structure:** embedding lookup is sparse (one row per token); block stack is block-diagonal across sequence positions except inside attention (dense $T \times T$ coupling per head).

## Warmup

With tiny config (vocab\_size=64, n\_ctx=16, n\_embd=32, n\_head=4, n\_layer=2):
Input `idx` of shape **(2, 8)**:

| Stage | Expression | Shape |
|-------|-----------|-------|
| Token embed | `transformer["wte"](idx)` | (2, 8, 32) |
| Pos indices | `arange(T, device=idx.device)` | (8,) |
| Pos embed | `transformer["wpe"](pos)` | (8, 32) → broadcast |
| Sum | `x = tok_emb + pos_emb` | (2, 8, 32) |
| Block 1 | `x = block1(x)` | (2, 8, 32) |
| Block 2 | `x = block2(x)` | (2, 8, 32) |
| Final norm | `x = ln_f(x)` | (2, 8, 32) |
| Logits | `F.linear(x, wte.weight)` | (2, 8, 64) |

## Your Task

Open `src/gpt2/model.py` and find `class GPT2Model`. Ensure the module imports `cast` from `typing` and `F` from `torch.nn.functional` (they are already present if you followed earlier tasks). Implement `forward`:

```python
def forward(self, idx: torch.Tensor) -> torch.Tensor:
    T = idx.size(1)
    assert T <= self.config.n_ctx, f"Sequence too long: {T} > {self.config.n_ctx}"

    # 1. Position indices on the same device as idx (avoid CPU→device copy)
    pos = torch.arange(T, dtype=torch.long, device=idx.device)

    # 2. Embeddings (ModuleDict bracket access)
    tok_emb = self.transformer["wte"](idx)   # (B, T, C)
    pos_emb = self.transformer["wpe"](pos)   # (T, C)  — broadcasts over B
    x = tok_emb + pos_emb                    # (B, T, C)

    # 3. Transformer blocks — cast silences "Module is not iterable" from Pyright
    for layer in cast(nn.ModuleList, self.transformer["h"]):
        x = layer(x)

    # 4. Final layer norm
    x = self.transformer["ln_f"](x)          # (B, T, C)

    # 5. Weight-tied logit projection (F.linear(x, W) ≡ x @ W.T)
    wte = cast(nn.Embedding, self.transformer["wte"])
    logits = F.linear(x, wte.weight)         # (B, T, vocab_size)
    return logits
```

**Do not** add `self.lm_head = nn.Linear(...)`. Weight tying reuses `wte.weight`.

**Prerequisites:** Tasks 04–08 must be complete before this test will pass.

## Example Input / Output

```python
from gpt2.config import GPT2_TINY
from gpt2.model import GPT2Model
import torch

model = GPT2Model(GPT2_TINY)
idx = torch.randint(0, 64, (2, 8))   # (B=2, T=8) with vocab_size=64
logits = model(idx)
print(logits.shape)    # torch.Size([2, 8, 64])

# Weight tying: no lm_head parameter, wte is used for projection
param_names = [n for n, _ in model.named_parameters()]
print(any("lm_head" in n for n in param_names))  # False

print(model.num_parameters())  # 28032 for tiny config (exact, not ~20K)
```

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `"Module" is not iterable` (Pyright) | `ModuleDict["h"]` typed as base `nn.Module` | `cast(nn.ModuleList, self.transformer["h"])` |
| Param-count test fails despite correct tying | Loose `<` bound ignores block params | Test uses exact architecture-derived count — see `_expected_param_count` in `tests/component/test_model_forward.py` |
| Duplicate ~38M params at GPT-2 Small | Separate `nn.Linear` lm\_head registered | Use `F.linear(x, wte.weight)` only |
| Slow forward on GPU | `arange(...).to(device)` each step | `arange(T, device=idx.device)` |

## Modern LLM Comparison

| Feature | GPT-2 | LLaMA 3 | Notes |
|---------|-------|---------|-------|
| Position embed | Learned wpe table | RoPE in attention | RoPE is applied inside attn |
| Weight tying | Yes (wte = lm\_head) | No (untied) | Untied can outperform at large scale |
| Post-embedding norm | None | RMSNorm after wte | Stabilises at large scale |
| Vocab padding | None | Padded to multiple of 64 | Helps tensor core utilisation |

## Run the Tests

```bash
uv run pytest tests/component/test_model_forward.py -v
```

The weight-tying tests are regression guards:

- `test_wte_and_lm_head_share_weights` — no separate `lm_head` parameter exists.
- `test_parameter_count_excludes_duplicate_lm_head` — total equals the exact tied count; an untied head would add `vocab_size * n_embd` params.
