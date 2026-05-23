# Guide 08 — Transformer Block

## Theory

One transformer block combines attention and MLP with **pre-norm residual connections**:

$$x \leftarrow x + \text{Attention}(\text{LayerNorm}_1(x))$$
$$x \leftarrow x + \text{MLP}(\text{LayerNorm}_2(x))$$

**Why pre-norm?**
In the original Transformer (Vaswani 2017) the norm came AFTER the residual add (post-norm). GPT-2 moved it before (pre-norm), following work by Xiong et al. (2020). The key benefit: at initialisation, the residual pathway passes the signal unchanged while the sub-layer branches start near zero. This gives a near-identity initialisation at any depth, making training stable without careful LR warm-up.

**Why residual connections?**
1. **Gradient highway**: gradients flow directly through the $+$ operation, bypassing the sub-layers. This prevents vanishing gradients in deep networks (He et al., ResNet, 2015).
2. **Ensemble interpretation**: a model with $N$ residual blocks implicitly represents an ensemble of $2^N$ subnetworks of different depths (Veit et al., 2016).

**The residual stream:**
A useful mental model: the residual stream $x$ carries information from the token embedding all the way to the logit head. Each block can **read from** and **write to** the stream via the attention and MLP sub-layers. The residual add is the mechanism for those writes.

## Warmup

Trace the computation for **one block** on input $x$ of shape $(B, T, C) = (2, 8, 32)$:

| Operation | Expression | Output shape |
|-----------|-----------|--------------|
| Pre-norm 1 | `self.ln_1(x)` | `(2, 8, 32)` |
| Attention | `self.attn(ln1_out)` | `(2, 8, 32)` |
| Residual 1 | `x = x + attn_out` | `(2, 8, 32)` |
| Pre-norm 2 | `self.ln_2(x)` | `(2, 8, 32)` |
| MLP | `self.mlp(ln2_out)` | `(2, 8, 32)` |
| Residual 2 | `x = x + mlp_out` | `(2, 8, 32)` |

The block is **shape-preserving**: input and output are always `(B, T, C)`.

## Your Task

Open `src/gpt2/model.py` and find `class Block`. Implement `forward`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```

This is two lines. The elegance of the pre-norm formulation is that it reads
directly like the mathematical definition.

**Prerequisites:** Tasks 04–07 must be complete before this test will pass.

## Example Input / Output

```python
from gpt2.config import GPT2_TINY
from gpt2.model import Block
import torch

block = Block(GPT2_TINY)
x = torch.randn(2, 8, 32)
y = block(x)
print(y.shape)              # torch.Size([2, 8, 32])
print(torch.allclose(y, x)) # False — the block modifies x
```

## Modern LLM Comparison

| Aspect | GPT-2 | LLaMA 3 | Notes |
|--------|-------|---------|-------|
| Norm position | Pre-norm | Pre-norm | Same |
| Norm type | LayerNorm | RMSNorm | RMSNorm cheaper |
| Attention | Full MHA | GQA | Different K/V head count |
| MLP | GELU 4× | SwiGLU 8/3× | Different activation + gate |
| Residual | Plain add | Plain add | Same |

The block structure itself is essentially unchanged from GPT-2 to LLaMA 3.
All the meaningful architectural improvements are inside the sub-layers.

## Run the Tests

```bash
uv run pytest tests/component/test_block.py -v
```

The gradient test verifies all 8 parameter tensors (ln\_1, attn, ln\_2, mlp)
receive gradients — a regression guard against forgetting a residual add.
