# Guide 10 — Language Modelling Loss

## Theory

GPT-2 is trained with **next-token prediction**: given tokens $t_1, \ldots, t_T$,
predict each next token $t_{i+1}$.

**The shift trick:** given a sequence $[t_0, t_1, t_2, t_3]$, we form:
- **inputs** (fed to the model): $[t_0, t_1, t_2]$
- **targets** (what to predict): $[t_1, t_2, t_3]$

One sequence gives $T-1$ supervised examples. This is why we do not need
explicit labels — the sequence supervises itself.

**Cross-entropy loss** for a single position $i$:

$$\mathcal{L}_i = -\log \frac{\exp(\text{logits}_{i, t_{i+1}})}{\sum_{v=1}^{V} \exp(\text{logits}_{i,v})} = -\log p_\theta(t_{i+1} \mid t_0, \ldots, t_i)$$

The total loss is the mean over all positions and batch elements:
$$\mathcal{L} = \frac{1}{B(T-1)} \sum_{b=1}^{B} \sum_{i=1}^{T-1} \mathcal{L}_{b,i}$$

**Expected loss at random initialisation:**
If the model outputs uniform logits, $p = 1/V$ for every token.
Loss = $-\log(1/V) = \log(V)$.
For GPT-2 Small ($V=50257$): $\log(50257) \approx 10.8$ nats.
For our tiny test vocab ($V=64$): $\log(64) \approx 4.16$ nats.

After training, a good GPT-2 Small achieves ~3.0 nats on web text.

## Warmup

Given sequence `"Hello World"` tokenised as `[15496, 2159, 995]`:

| Position | Input token | Target token |
|----------|------------|-------------|
| 0 | 15496 ("Hello") | 2159 (" World") |
| 1 | 2159 (" World") | 995 ("!") |

Q: For sequence $[A, B, C, D]$ with $T=4$, write out all (input, target) pairs.
Answer: (A→B), (B→C), (C→D) — three pairs from four tokens.

Q: Why do we include `logits[:, :-1, :]` and `targets[:, 1:]`? Why not just use the full sequence?
A: The last logit has no target (there's no $T+1$ token), so we drop it. The first target has no preceding context logit at position -1, so we start targets from index 1.

## Your Task

Open `src/gpt2/train.py` and find `cross_entropy_loss`. Implement it:

```python
def cross_entropy_loss(logits, token_ids):
    # logits:    (B, T, vocab_size)
    # token_ids: (B, T)

    # Shift: model predicts token at position t given 0..t-1
    logits_shifted  = logits[:, :-1, :].contiguous()   # (B, T-1, V)
    targets_shifted = token_ids[:, 1:].contiguous()     # (B, T-1)

    # Flatten and compute cross-entropy
    return F.cross_entropy(
        logits_shifted.view(-1, logits.size(-1)),  # (B*(T-1), V)
        targets_shifted.view(-1),                  # (B*(T-1),)
    )
```

## Example Input / Output

```python
from gpt2.config import GPT2_TINY
from gpt2.model import GPT2Model
from gpt2.train import cross_entropy_loss
import torch, math

model = GPT2Model(GPT2_TINY)
idx = torch.randint(0, 64, (2, 8))
logits = model(idx)
loss = cross_entropy_loss(logits, idx)

print(loss.shape)   # torch.Size([])  ← scalar
print(loss.item())  # ≈ log(64) ≈ 4.16 at random init
print(math.log(64)) # 4.158...  ← theoretical expected value
```

## Modern LLM Comparison

| Aspect | GPT-2 | Modern LLMs | Notes |
|--------|-------|-------------|-------|
| Loss function | Cross-entropy | Cross-entropy | Unchanged |
| Target | Next token | Next token | Unchanged |
| Label smoothing | None | Sometimes | Rarely used for LLMs |
| Chunked CE | No | Yes (in practice) | Avoid materialising (B,T,V) logits for large V |

**Chunked cross-entropy** avoids holding the full `(B, T, V)` logit tensor in
memory at once. For GPT-2 Small ($V=50257$) this matrix is manageable, but for
modern models with $V=128256$ and large $T$, it can use tens of GB.

## Run the Tests

```bash
uv run pytest tests/system/test_training_smoke.py -k loss -v
```
