# Guide 13 — Text Generation

## Theory

### Autoregressive Generation

The model generates one token at a time:

```
context = [t₀, t₁, ..., t_{T-1}]
for each new step:
    logits = model(context)               # shape (1, T, V)
    logits_last = logits[:, -1, :]        # shape (1, V) — only last position
    sample next_token from logits_last
    context = [t₀, ..., t_{T-1}, next_token]
```

Only the **last position's** logits are used because position $T-1$ has access
to all previous tokens and is the only one predicting the next token.

### Temperature

$$\text{logits\_scaled} = \text{logits} / \tau$$

- $\tau = 1.0$: unchanged distribution
- $\tau < 1.0$: sharper (more confident), less diverse
- $\tau \to 0$: greedy decoding (argmax), fully deterministic
- $\tau > 1.0$: flatter, more random/creative

### Top-k Filtering

Keep only the $k$ tokens with the highest logits; set the rest to $-\infty$.

For logits $[1.0, 2.0, 0.5, 3.0]$ with $k=2$:
- Top-2 values: $3.0$ (index 3) and $2.0$ (index 1)
- After filtering: $[-\infty, 2.0, -\infty, 3.0]$

### Top-p (Nucleus) Sampling

Sort tokens by probability descending; keep the smallest set whose cumulative
probability $\geq p$:

For probs $[0.5, 0.3, 0.15, 0.04, 0.01]$ with $p=0.9$:
- Cumulative: $[0.5, 0.8, 0.95, 0.99, 1.0]$
- First index where cum $\geq 0.9$: index 2 (value 0.95)
- Keep indices 0, 1, 2; remove 3 and 4

Top-p is adaptive: when the model is uncertain (flat distribution), the nucleus
is large (many tokens sampled); when confident (peaked), the nucleus shrinks.

## Warmup

1. **Top-k:** For logits $[0.1, 0.5, 0.9, 0.2]$ and $k=1$:
   Which token survives? What is its logit?

2. **Top-p:** For probabilities $[0.7, 0.2, 0.05, 0.03, 0.02]$ and $p=0.9$:
   - Cumulative: $[0.7, 0.9, 0.95, 0.98, 1.0]$
   - Which index is the first to reach $\geq 0.9$? (index 1)
   - How many tokens survive?

3. **Autoregressive loop:** With context length T=3 and max\_new\_tokens=4:
   After step 1: context length = ?  After step 4: context length = ?

## Your Tasks

Open `src/gpt2/generate.py` and implement three functions:

**Task 13a — `top_k_filter`:**
```python
values, _ = torch.topk(logits, k, dim=-1)
threshold = values[..., -1:]   # the k-th largest value
return logits.masked_fill(logits < threshold, float('-inf'))
```

**Task 13b — `top_p_filter`:**
```python
sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
sorted_probs = F.softmax(sorted_logits, dim=-1)
cum_probs = torch.cumsum(sorted_probs, dim=-1)
# Remove tokens where cum_prob BEFORE this token already exceeds p
remove = (cum_probs - sorted_probs) > p
sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
return sorted_logits.scatter(-1, sorted_idx, sorted_logits)
```

**Task 13c — `generate`:**
```python
for _ in range(max_new_tokens):
    idx_crop = idx[:, -model.config.n_ctx:]       # don't exceed context window
    logits = model(idx_crop)[:, -1, :]            # (B, V)
    if temperature == 0.0:
        next_id = logits.argmax(dim=-1, keepdim=True)
    else:
        logits = logits / temperature
        if top_k is not None: logits = top_k_filter(logits, top_k)
        if top_p is not None: logits = top_p_filter(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
    idx = torch.cat([idx, next_id], dim=1)
return idx
```

## Example Input / Output

```python
from gpt2.config import GPT2_TINY
from gpt2.model import GPT2Model
from gpt2.generate import generate
import torch

model = GPT2Model(GPT2_TINY)
model.eval()
prompt = torch.tensor([[5, 12, 7]])   # shape (1, 3)

out = generate(model, prompt, max_new_tokens=5, temperature=0.0)
print(out.shape)   # torch.Size([1, 8])  (3 + 5)
print(out[0, :3])  # tensor([5, 12, 7])  ← prompt preserved

# Same seed → same output
out1 = generate(model, prompt, max_new_tokens=5, seed=42)
out2 = generate(model, prompt, max_new_tokens=5, seed=42)
assert torch.equal(out1, out2)
```

## Modern LLM Comparison

| Feature | This implementation | Production |
|---------|--------------------|-----------| 
| KV cache | No | Yes (O(T) per step vs O(T²)) |
| Speculative decoding | No | Yes (2-3× throughput) |
| Beam search | No | Sometimes for structured output |
| Batched generation | Yes (B > 1) | Yes, with padding |
| top-k + top-p combined | Yes | Yes (both applied in order) |

**KV cache** (not implemented here): instead of recomputing $K$ and $V$ for
all past tokens at every step, store them after the first forward pass and
reuse. Each new step only computes new $Q$ and appends new $K, V$. This
reduces cost per step from $O(T^2)$ to $O(T)$.

## Run the Tests

```bash
uv run pytest tests/component/test_generate.py -v
```
