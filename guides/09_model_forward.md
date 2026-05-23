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
$$\text{logits} = h \cdot \text{wte.weight}^\top \in \mathbb{R}^{B \times T \times V}$$

**Weight tying** (Press & Wolf 2017): the output projection matrix is the
*same tensor* as the token embedding matrix `wte.weight`. This halves the
embedding parameter count (~38M for GPT-2 Small) and often improves
perplexity by encouraging consistent token representations throughout the model.

## Warmup

With tiny config (vocab\_size=64, n\_ctx=16, n\_embd=32, n\_head=4, n\_layer=2):
Input `idx` of shape **(2, 8)**:

| Stage | Expression | Shape |
|-------|-----------|-------|
| Token embed | `wte(idx)` | (2, 8, 32) |
| Pos indices | `arange(0, 8)` | (8,) |
| Pos embed | `wpe(pos)` | (8, 32) → broadcast |
| Sum | `h = tok_emb + pos_emb` | (2, 8, 32) |
| Block 1 | `h = block1(h)` | (2, 8, 32) |
| Block 2 | `h = block2(h)` | (2, 8, 32) |
| Final norm | `h = ln_f(h)` | (2, 8, 32) |
| Logits | `h @ wte.weight.T` | (2, 8, 64) |

## Your Task

Open `src/gpt2/model.py` and find `class GPT2Model`. Implement `forward`:

```python
def forward(self, idx: torch.Tensor) -> torch.Tensor:
    B, T = idx.size()
    assert T <= self.config.n_ctx, f"Sequence too long: {T} > {self.config.n_ctx}"

    # 1. Position indices (0, 1, ..., T-1) on the same device as idx
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

    # 2. Embeddings
    tok_emb = self.transformer.wte(idx)   # (B, T, C)
    pos_emb = self.transformer.wpe(pos)   # (T, C)  — broadcasts over B
    h = tok_emb + pos_emb                 # (B, T, C)

    # 3. Transformer blocks
    for block in self.transformer.h:
        h = block(h)

    # 4. Final layer norm
    h = self.transformer.ln_f(h)          # (B, T, C)

    # 5. Weight-tied logit projection
    logits = h @ self.transformer.wte.weight.T   # (B, T, vocab_size)
    return logits
```

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

print(model.num_parameters())  # ~20K for tiny config
```

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

The weight-tying test is a regression guard: if you accidentally add an
`nn.Linear` lm\_head, the test fails and warns you of the duplicate parameters.
