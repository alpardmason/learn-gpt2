# Guide 05 — Layer Normalization

## Theory

**Layer Normalisation** (Ba et al., 2016) normalises the activations across
the feature dimension (last axis) for each token independently:

$$\mu = \frac{1}{C}\sum_{i=1}^{C} x_i \qquad \sigma^2 = \frac{1}{C}\sum_{i=1}^{C}(x_i - \mu)^2$$

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \qquad \text{out} = \gamma \odot \hat{x} + \beta$$

- $\gamma$ (weight) and $\beta$ (bias) are **learnable** vectors of size `n_embd`, initialised to 1 and 0
- $\varepsilon = 10^{-5}$ prevents division by zero

**Why LayerNorm, not BatchNorm?**
BatchNorm normalises across the **batch** dimension, which requires a large
batch size to get stable statistics and doesn't work well for variable-length
sequences. LayerNorm normalises across the **feature** dimension — it only
looks at a single token, so it works equally well with batch size 1.

**Pre-norm vs Post-norm (critical):**
GPT-2 applies LayerNorm **before** each sub-layer (pre-norm):
```
x = x + Attention(LayerNorm(x))   # normalize THEN attend
x = x + MLP(LayerNorm(x))         # normalize THEN MLP
```
The original Transformer paper used post-norm (after the residual add).
Pre-norm is much more stable for deep networks because it guarantees the
input to each sub-layer has unit variance, regardless of depth.

## Warmup

Normalise the vector $[1.0, 2.0, 3.0]$ by hand (assume $\varepsilon = 0$, $\gamma = [1,1,1]$, $\beta = [0,0,0]$):

$$\mu = \frac{1+2+3}{3} = 2.0$$
$$\sigma^2 = \frac{(1-2)^2 + (2-2)^2 + (3-2)^2}{3} = \frac{1+0+1}{3} = \frac{2}{3}$$
$$\sqrt{\sigma^2} = \sqrt{2/3} \approx 0.8165$$
$$\hat{x} = \frac{[1-2,\; 2-2,\; 3-2]}{0.8165} \approx [-1.2247, \; 0.0, \; 1.2247]$$

Verify: the output has mean = 0 and variance = 1. ✓

## Your Task

Open `src/gpt2/model.py` and find `class LayerNorm`. Implement `forward`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Steps:
    # 1. mean = x.mean(dim=-1, keepdim=True)
    # 2. var  = x.var(dim=-1, keepdim=True, unbiased=False)
    # 3. x_hat = (x - mean) / (var + self.eps).sqrt()
    # 4. return self.weight * x_hat + self.bias
    raise NotImplementedError(...)
```

**Important:** use `unbiased=False` in `x.var()` — this is the population
variance (divides by N), not the sample variance (divides by N-1). This
matches the LayerNorm paper and PyTorch's `nn.LayerNorm`.

## Example Input / Output

```python
from gpt2.model import LayerNorm
import torch

ln = LayerNorm(32)
x = torch.randn(2, 8, 32)  # (B, T, C)
y = ln(x)

print(y.shape)                      # torch.Size([2, 8, 32])
print(y.mean(dim=-1).abs().max())   # ≈ 0.0  (normalised to mean 0)
print(y.var(dim=-1).mean())         # ≈ 1.0  (normalised to var 1)
# Note: var won't be exactly 1 because of the learnable weight γ (init=1)
```

## Modern LLM Comparison

| Model | Normalisation | Formula |
|-------|--------------|---------|
| GPT-2 | LayerNorm | $(x - \mu) / \sqrt{\sigma^2 + \varepsilon}$ then $\gamma, \beta$ |
| LLaMA, Mistral, Gemma | RMSNorm | $x / \sqrt{\text{mean}(x^2) + \varepsilon}$ then $\gamma$ (no $\beta$) |
| Original Transformer | LayerNorm (post-norm) | Same formula, different position |

**RMSNorm** drops the mean subtraction. This saves ~10% compute (no mean
calculation) and works equally well empirically. It has no bias parameter ($\beta$).

## Run the Tests

```bash
uv run pytest tests/unit/test_layernorm.py -v
```

Expected: 7 tests pass, including zero-mean/unit-variance checks and agreement
with `torch.nn.LayerNorm`.
