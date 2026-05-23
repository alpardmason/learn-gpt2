# Guide 07 — MLP (Feed-Forward Network)

## Theory

The MLP in each transformer block is a **position-wise** two-layer feed-forward
network: it processes each token's embedding independently (no information
exchange between positions, unlike attention).

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

- $W_1 \in \mathbb{R}^{C \times 4C}$: expands from `n_embd` → `4 * n_embd`
- $W_2 \in \mathbb{R}^{4C \times C}$: contracts back to `n_embd`
- The **4× expansion factor** is a design choice from the original Transformer
  paper. The inner dimension stores "working memory" for transformations.

**Why does the MLP matter?**
Attention is good at routing information (which tokens to mix), but the MLP
is where most of the "knowledge storage" happens. Factual recall experiments
show that many facts can be associated with specific MLP weights (Meng et al.,
ROME 2022).

**Parameter count for MLP in one block:**
$2 \times C \times 4C = 8C^2$. For GPT-2 Small (C=768): $8 \times 768^2 \approx 4.7M$ per block, $\times 12 = 56M$ total MLP params.

## Warmup

1. With `n_embd = 32`, compute the weight shapes of `c_fc` and `c_proj`:
   - `c_fc.weight`:  shape = `(4×32, 32)` = `(128, 32)` ✓ *(PyTorch stores as (out, in))*
   - `c_proj.weight`: shape = `(32, 128)` ✓

2. An input tensor `x` has shape `(2, 8, 32)`. Trace the shape through each step:
   - After `c_fc`:   shape = ?
   - After `GELU`:   shape = ?
   - After `c_proj`: shape = ?
   
   Answer: (2,8,128) → (2,8,128) → (2,8,32)

3. Why is a bias term included in the linear layers? (Hint: LayerNorm's bias can be set to 0; what would happen to the model's expressiveness?)

## Your Task

Open `src/gpt2/model.py` and find `class MLP`. Implement `forward`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Three lines:
    x = self.c_fc(x)    # (B, T, C) → (B, T, 4C)
    x = self.gelu(x)    # apply GELU elementwise
    x = self.c_proj(x)  # (B, T, 4C) → (B, T, C)
    return x
```

This is the simplest implementation task in the project.

## Example Input / Output

```python
from gpt2.config import GPT2_TINY
from gpt2.model import MLP
import torch

mlp = MLP(GPT2_TINY)
x = torch.randn(2, 8, 32)   # (B=2, T=8, C=32)
y = mlp(x)
print(y.shape)               # torch.Size([2, 8, 32])  ← shape preserved

print(mlp.c_fc.weight.shape)    # torch.Size([128, 32])
print(mlp.c_proj.weight.shape)  # torch.Size([32, 128])
```

## Modern LLM Comparison

| Model | FFN Type | Inner dim | Weight matrices |
|-------|---------|-----------|----------------|
| GPT-2 | 2-layer GELU | 4C | 2 (W₁, W₂) |
| LLaMA, Mistral, Gemma | SwiGLU | ~8/3 · C | 3 (W₁, W₂, W₃) |
| Mixtral | SwiGLU + MoE | ~8/3 · C | N experts × 3 |

**SwiGLU:**
$$\text{FFN}_{\text{SwiGLU}}(x) = \bigl(\text{SiLU}(xW_1) \odot xW_3\bigr) W_2$$

The gating mechanism ($\odot$ elementwise multiplication) gives the network a
way to suppress certain features adaptively. Despite 3 weight matrices, the
expansion is only ~8/3× (not 4×), keeping parameter counts comparable to GPT-2.

## Run the Tests

```bash
uv run pytest tests/unit/test_mlp.py -v
```
