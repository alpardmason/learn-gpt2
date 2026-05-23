# Guide 12 — Training Loop

## Theory

### AdamW

GPT-2 is trained with **AdamW** (Loshchilov & Hutter, 2019), which fixes a subtle
bug in L2-regularised Adam by decoupling weight decay from the adaptive moment updates:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(1st moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(2nd moment)}$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t) \qquad \text{(bias correction)}$$
$$\theta_t = \theta_{t-1} - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda \theta_{t-1}\right)$$

GPT-2 hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, $\lambda = 0.01$.

**Weight decay** ($\lambda$) is only applied to weight matrices (2-D parameters),
NOT to biases or LayerNorm parameters (1-D). The `Trainer` class sets this up
correctly via two parameter groups.

### Gradient Clipping

$$g \leftarrow g \cdot \min\!\left(1,\ \frac{\tau}{\|g\|_2}\right) \quad \tau = 1.0$$

If the global L2 norm of all gradient tensors combined exceeds 1.0, they are
rescaled to have norm exactly 1.0. This is a safety net against the occasional
exploding gradient that can occur early in training.

### Training Step Order

This order is mandatory. Any deviation causes subtle bugs:

```
1. optimizer.zero_grad(set_to_none=True)   # clear stale gradients
2. logits = model(token_ids)               # forward pass
3. loss = cross_entropy_loss(logits, token_ids)
4. loss.backward()                         # backward pass
5. torch.nn.utils.clip_grad_norm_(params, 1.0)  # clip
6. lr = _update_lr()                       # update LR in all param groups
7. optimizer.step()                        # update weights
8. self.step += 1
```

## Warmup

Order these operations correctly (fill in 1–6):

| Step | Operation |
|------|-----------|
| ___ | `loss.backward()` |
| ___ | `optimizer.zero_grad(set_to_none=True)` |
| ___ | `optimizer.step()` |
| ___ | `logits = model(idx)` |
| ___ | `loss = cross_entropy_loss(logits, idx)` |
| ___ | `clip_grad_norm_(params, 1.0)` |

**Answer:** 4, 1, 6, 2, 3, 5

**Why `set_to_none=True`?** It frees the gradient tensors instead of filling
them with zeros. For models with many parameters, this saves significant memory.

## Your Task

Open `src/gpt2/train.py` and find `Trainer.train_step`. Implement it:

```python
def train_step(self, token_ids):
    self.optimizer.zero_grad(set_to_none=True)
    logits = self.model(token_ids)
    loss = cross_entropy_loss(logits, token_ids)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
    self._update_lr()
    self.optimizer.step()
    self.step += 1
    return loss.item()
```

**Prerequisites:** Tasks 09, 10, and 11 must be complete.

## Example Input / Output

After 50 steps on a repeating sequence, loss should decrease by ≥10%:

```
step     0 | loss 4.1583  (≈ log(64), random init)
step    10 | loss 3.8921
step    20 | loss 3.2314
step    30 | loss 2.1455
step    40 | loss 1.3219
step    49 | loss 0.8712  (model is memorising the pattern)
```

## Modern LLM Comparison

| Feature | This implementation | Production LLM training |
|---------|--------------------|-----------------------|
| Precision | fp32 | bf16 forward + fp32 master weights |
| Gradient accumulation | No (batch=1 per step) | Yes (simulate large batches) |
| Distributed training | No | DDP or FSDP across many GPUs |
| Optimizer | AdamW | AdamW or Muon (newer) |
| Gradient checkpointing | No | Often yes (saves activation memory) |
| Batch size | 4–8 sequences | 512–4096 sequences |

## Run the Tests

```bash
uv run pytest tests/system/test_training_smoke.py -v
```

The key test: `test_loss_decreases_on_repeating_sequence` runs 50 steps and
asserts `final_loss < initial_loss * 0.9`. This is your final end-to-end check.
