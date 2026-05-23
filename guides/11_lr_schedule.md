# Guide 11 — Learning Rate Schedule

## Theory

GPT-2 uses a **cosine annealing schedule with linear warmup** (§3 of the paper):

**Phase 1 — Linear warmup** (steps $0 \to T_{\text{warm}}$):
$$\text{lr}(t) = \text{lr\_max} \cdot \frac{t}{T_{\text{warm}}}$$

**Phase 2 — Cosine decay** (steps $T_{\text{warm}} \to T_{\text{max}}$):
$$\text{progress} = \frac{t - T_{\text{warm}}}{T_{\text{max}} - T_{\text{warm}}} \in [0, 1]$$
$$\text{lr}(t) = \text{lr\_min} + \frac{1}{2}(\text{lr\_max} - \text{lr\_min}) \cdot (1 + \cos(\pi \cdot \text{progress}))$$

**Phase 3** (after $T_{\text{max}}$): $\text{lr}(t) = \text{lr\_min}$

The GPT-2 paper sets $\text{lr\_min} = \text{lr\_max} / 10$.

**Why warmup?**
At step 0, the model's parameters are randomly initialised. The gradient
estimates are high-variance and unreliable. A low learning rate in the first
few hundred steps prevents large destructive updates while the model settles.

**Why cosine?**
The cosine curve gives a smooth transition from peak to minimum LR, avoiding
the hard drop of step-based schedules. The half-cosine form ($(1 + \cos(\pi t)) / 2$)
exactly covers $[1.0, 0.0]$ as $t$ goes from $0$ to $1$.

## Warmup

Sketch the LR curve for $\text{lr\_max} = 10^{-3}$, $\text{lr\_min} = 10^{-4}$, $T_{\text{warm}} = 100$, $T_{\text{max}} = 1000$:

| Step | Phase | lr (approx) |
|------|-------|------------|
| 0 | warmup | 0.0 |
| 50 | warmup | 5×10⁻⁴ |
| 100 | end of warmup | 10⁻³ |
| 550 | cosine (progress=0.5) | ~5.5×10⁻⁴ |
| 1000 | end of cosine | 10⁻⁴ |
| 9999 | after end | 10⁻⁴ (clipped) |

Verify: at progress=0.5, $\cos(\pi/2) = 0$, so $\text{lr} = \text{lr\_min} + 0.5 \cdot (\text{lr\_max} - \text{lr\_min}) = 5.5 \times 10^{-4}$ ✓

## Your Task

Open `src/gpt2/train.py` and find `cosine_lr_schedule`. Implement it:

```python
def cosine_lr_schedule(step, max_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps           # linear warmup

    if step <= max_steps:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + coeff * (max_lr - min_lr)     # cosine decay

    return min_lr                                     # after max_steps
```

## Example Input / Output

```python
from gpt2.train import cosine_lr_schedule

lr = cosine_lr_schedule
print(lr(0,    max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000))  # 0.0
print(lr(100,  max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000))  # 1e-3
print(lr(1000, max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000))  # 1e-4
print(lr(9999, max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000))  # 1e-4
```

## Modern LLM Comparison

| Schedule | Models | Notes |
|----------|--------|-------|
| Cosine + warmup | GPT-2, GPT-3, PaLM | Classic, well-understood |
| WSD (Warmup-Stable-Decay) | MiniCPM, Falcon | Constant LR then steep decay; enables extending training |
| Cosine with restarts | Some | Warm restarts allow escaping local minima |
| Constant | Debugging | Never use for real training |

**Warmup-Stable-Decay (WSD):** train at constant $\text{lr\_max}$ for most of
training, then decay steeply at the end. The advantage: you can continue
training from a checkpoint without re-running the entire cosine decay from scratch.

## Run the Tests

```bash
uv run pytest tests/system/test_training_smoke.py::TestCosineSchedule -v
```
