# Guide 02 — Configuration (`GPT2Config`)

## Theory

GPT-2 comes in four sizes (Table 2, GPT-2 paper). All share the same
architecture but differ in depth and width:

| Size | n\_layer | n\_embd | n\_head | ~Params |
|------|---------|--------|--------|---------|
| Small | 12 | 768 | 12 | 117M |
| Medium | 24 | 1024 | 16 | 345M |
| Large | 36 | 1280 | 20 | 762M |
| XL | 48 | 1600 | 25 | 1542M |

**Key relationship:** `head_dim = n_embd / n_head`. Each attention head operates
on a `head_dim`-dimensional subspace of the residual stream. For GPT-2 Small:
`head_dim = 768 / 12 = 64`.

**Parameter count (excluding embeddings):**
$$N \approx 12 \cdot n\_{\text{layer}} \cdot n\_{\text{embd}}^2$$

For Small: $12 \times 12 \times 768^2 \approx 85M$ (the rest is embeddings).

## Warmup

1. Using the formula above, estimate the non-embedding parameter count for
   GPT-2 Medium (n\_layer=24, n\_embd=1024). Compare to the official 345M.
2. Why must `n_embd` be divisible by `n_head`? What breaks if it isn't?

## Your Task

`src/gpt2/config.py` is **already fully implemented** — no stubs here.
Your task is to read and understand it:

1. Find `GPT2Config`. Note it is a `frozen=True` dataclass. What does frozen mean?
2. Find `GPT2_SMALL`, `GPT2_MEDIUM`, `GPT2_LARGE`, `GPT2_XL`. Verify they
   match the paper's Table 2.
3. Find `GPT2_TINY`. This is used in all tests. Note its tiny dimensions.

## Example Input / Output

```python
from gpt2.config import GPT2Config, GPT2_SMALL, GPT2_TINY

cfg = GPT2_SMALL
print(cfg.n_embd)    # 768
print(cfg.head_dim)  # 64  (computed property: 768 // 12)

tiny = GPT2_TINY
print(tiny.vocab_size)  # 64
print(tiny.n_ctx)       # 16
print(tiny.head_dim)    # 8   (32 // 4)
```

## Modern LLM Comparison

| Parameter | GPT-2 | LLaMA 3 (8B) | GPT-4 (rumoured) |
|-----------|-------|-------------|-----------------|
| n\_layer | 12–48 | 32 | ~96 |
| n\_embd | 768–1600 | 4096 | ~12288 |
| n\_head | 12–25 | 32 | ~96 |
| head\_dim | 64 | 128 | 128 |
| vocab\_size | 50257 | 128256 | ~100K |

Modern models increase head\_dim from 64 to 128, which improves FlashAttention
efficiency (better tile utilisation).

## Run the Tests

```bash
uv run pytest tests/unit/test_config.py -v
```
