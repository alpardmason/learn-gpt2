# Guide 04 — GELU Activation

## Theory

**GELU** (Gaussian Error Linear Unit, Hendrycks & Gimpel 2016) is defined as:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the standard Gaussian CDF. Intuitively: GELU scales the
input by the probability that a standard Gaussian sample is less than $x$.
This creates a **soft gate**: when $x$ is large and positive, the gate ≈ 1
(pass through); when $x$ is large and negative, the gate ≈ 0 (suppress).

**Tanh approximation** (what GPT-2 uses):

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}} \cdot (x + 0.044715 x^3)\right)\right)$$

This approximation is accurate to within $10^{-4}$ for all $x$ and is
differentiable everywhere — no hard cutoff like ReLU.

**Why not ReLU?**
- ReLU has a hard zero for $x < 0$: dead neurons can permanently stop
  contributing (if their pre-activation stays negative throughout training).
- GELU has a small non-zero gradient for slightly negative inputs, allowing
  recovery. It consistently outperforms ReLU on language tasks.

## Warmup

Compute by hand using the tanh formula. Let $\alpha = \sqrt{2/\pi} \approx 0.7979$:

1. **GELU(0.0):**
   $\tanh(\alpha \cdot (0 + 0)) = \tanh(0) = 0$
   $\text{GELU}(0) = 0.5 \cdot 0 \cdot (1 + 0) = \mathbf{0.0}$

2. **GELU(1.0):**
   Inner: $\alpha \cdot (1 + 0.044715) \approx 0.7979 \times 1.0447 \approx 0.8337$
   $\tanh(0.8337) \approx 0.6832$
   $\text{GELU}(1) = 0.5 \times 1 \times (1 + 0.6832) \approx \mathbf{0.8416}$

3. **Sketch the curve:** At what $x$ does GELU cross $y = x$ (i.e. gate = 1)?

## Your Task

Open `src/gpt2/model.py` and find `class GELU`. Implement `forward`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Implement: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    raise NotImplementedError(...)
```

**Useful constants:**
```python
import math
math.sqrt(2.0 / math.pi)  # ≈ 0.7978845608
```

**Useful PyTorch functions:**
```python
torch.tanh(x)
x ** 3  # or x.pow(3)
```

## Example Input / Output

With the tiny test config (run in a Python REPL after implementing):

```python
from gpt2.model import GELU
import torch

gelu = GELU()
print(gelu(torch.tensor(0.0)).item())   # 0.0
print(gelu(torch.tensor(0.5)).item())   # ≈ 0.3457
print(gelu(torch.tensor(1.0)).item())   # ≈ 0.8413
print(gelu(torch.tensor(-1.0)).item())  # ≈ -0.1587

# Batch of shape (B, T, C) → same shape
x = torch.randn(2, 8, 32)
y = gelu(x)
print(y.shape)  # torch.Size([2, 8, 32])
```

## Modern LLM Comparison

| Model | Activation | Notes |
|-------|-----------|-------|
| GPT-2 | GELU (tanh approx.) | This task |
| BERT | GELU (exact) | Slower, same quality |
| LLaMA 2 / 3 | SwiGLU | `SiLU(xW₁) ⊙ (xW₂)` — gated, 3 weight matrices |
| GPT-NeoX | GELU | Same as GPT-2 |

**SwiGLU** (Shazeer 2020) is strictly better on benchmarks but requires 3 weight
matrices instead of 2 in the MLP, complicating the architecture.

## Run the Tests

```bash
uv run pytest tests/unit/test_gelu.py -v
```

Expected: 6 tests pass, including numerical value checks and agreement with
`torch.nn.GELU(approximate='tanh')`.
