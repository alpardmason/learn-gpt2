# Guide 06 — Causal Multi-Head Self-Attention

## Theory

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

- $Q, K, V \in \mathbb{R}^{T \times d_k}$ (queries, keys, values)
- $\sqrt{d_k}$ scaling: without it, dot products grow large for high $d_k$, pushing softmax into saturation (near-zero gradients)
- Softmax produces attention weights: non-negative, sum to 1 across the key dimension
- The output is a weighted sum of values: token $i$ aggregates information from all tokens it attends to

### Multi-Head Attention

Instead of one attention function, run $H$ in parallel on $d_k = C/H$ dimensional subspaces:

$$\text{head}_i = \text{Attention}(x W_i^Q,\ x W_i^K,\ x W_i^V)$$
$$\text{MultiHead}(x) = \text{concat}(\text{head}_1, \ldots, \text{head}_H) W^O$$

Each head can specialise: one head might attend to syntactic dependencies,
another to semantic relationships, etc.

**In practice:** a single QKV projection (`c_attn`) of size $3C$ followed by a
reshape is equivalent to $H$ separate projections. GPT-2 uses this efficient formulation.

### Causal Mask

GPT-2 is an **autoregressive** language model: token $i$ may only attend to
tokens $j \leq i$. We enforce this by adding $-\infty$ to positions $j > i$
in the attention logits before softmax, making those weights exactly 0.

```
Attention weights for T=4 (after masking, before softmax):
       pos0   pos1   pos2   pos3
pos0 [  0      -∞     -∞     -∞  ]
pos1 [  0       0     -∞     -∞  ]
pos2 [  0       0      0     -∞  ]
pos3 [  0       0      0      0  ]
```

## Warmup

1. **Draw the causal mask** for T=5 (5×5 matrix). Which cells are allowed (0) and which are masked (−∞)?

2. **Dimension check:** With n\_embd=32, n\_head=4:
   - What is `head_dim`? 
   - After the QKV projection, what is the shape of Q (before reshaping)?
   - After reshaping, what is the shape of Q?
   
   Answer: head\_dim=8; Q shape (B,T,32); after reshape (B,4,T,8).

3. **Scale:** Why do we divide by $\sqrt{d_k}$ and not by $d_k$ itself?

## Your Task

Open `src/gpt2/model.py` and find `class CausalSelfAttention`. Implement `forward`.

Step-by-step shape guide (B=2, T=8, C=32, H=4, D=8):

```python
# Step 1: QKV projection
qkv = self.c_attn(x)        # (B, T, 3C) = (2, 8, 96)

# Step 2: Split
q, k, v = qkv.split(self.n_embd, dim=2)  # each (B, T, C) = (2, 8, 32)

# Step 3: Reshape to separate heads
q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D) = (2,4,8,8)
k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # same
v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # same

# Step 4: Causal scaled dot-product attention
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # (2,4,8,8)

# Step 5: Merge heads
y = y.transpose(1, 2).contiguous().view(B, T, C)               # (2, 8, 32)

# Step 6: Output projection
y = self.c_proj(y)                                              # (2, 8, 32)
```

## Example Input / Output

```python
from gpt2.config import GPT2_TINY
from gpt2.model import CausalSelfAttention
import torch

attn = CausalSelfAttention(GPT2_TINY)
x = torch.randn(2, 8, 32)   # (B=2, T=8, C=32)
y = attn(x)
print(y.shape)               # torch.Size([2, 8, 32])
```

Causal mask test:
```python
# Perturbing the last token should NOT affect earlier positions
x1 = torch.randn(1, 6, 32)
x2 = x1.clone()
x2[:, -1, :] = torch.randn(1, 32)   # change the last token
y1, y2 = attn(x1), attn(x2)
torch.allclose(y1[:, :-1, :], y2[:, :-1, :])  # → True
```

## Modern LLM Comparison

| Feature | GPT-2 | LLaMA 3 | Notes |
|---------|-------|---------|-------|
| Positional encoding | Learned absolute (wpe) | RoPE | RoPE encodes relative positions in Q,K |
| Attention heads | Full MHA (H=H\_KV) | GQA (H\_KV < H) | GQA reduces KV cache size |
| Attention kernel | PyTorch SDPA | FlashAttention 3 | FA3 avoids materialising T×T matrix |
| QK norm | No | Yes (some models) | Stabilises training at large scale |

**RoPE** rotates Q and K by a position-dependent angle, encoding relative distances
without needing a separate position embedding table. This generalises better to
sequence lengths longer than seen during training.

## Run the Tests

```bash
uv run pytest tests/unit/test_attention.py -v
```

The causal mask test is the most important one — pass it before moving on.
