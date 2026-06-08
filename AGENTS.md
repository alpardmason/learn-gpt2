# AGENTS.md — learn-gpt2

Reference document for AI agents and developers working on this codebase.

---

## Tech Stack & Environment

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ | Runtime |
| uv | latest | Venv + package management |
| PyTorch | ≥ 2.6.0 | Tensor ops, autograd, nn.Module |
| tiktoken | ≥ 0.7.0 | BPE tokenizer |
| pytest | ≥ 8.0.0 | Test runner (autograder) |
| ruff | ≥ 0.6.0 | Linter (PEP8 + imports + modern Python) |

**Setup:**
```bash
uv venv --python 3.12
uv sync --dev
```

**Run tests:**
```bash
uv run pytest tests/unit/ -v              # fast (no GPU)
uv run pytest tests/ -m "not slow" -v    # all fast
uv run pytest tests/ -v                  # full suite
```

**Lint:**
```bash
uv run ruff check src/ tests/
```

---

## Architecture Summary

```
src/gpt2/
  config.py    GPT2Config(frozen dataclass), GPT2_SMALL/MEDIUM/LARGE/XL/TINY
  tokenizer.py CharTokenizer (complete), BPETokenizer (stub wrapping tiktoken)
  model.py     GELU → LayerNorm → CausalSelfAttention → MLP → Block → GPT2Model
  train.py     cross_entropy_loss, cosine_lr_schedule, TrainConfig, Trainer
  generate.py  top_k_filter, top_p_filter, generate
```

The model uses:
- **Pre-norm residual blocks** (LayerNorm before each sub-layer)
- **Learned absolute position embeddings** (wpe, shape `n_ctx × n_embd`)
- **Weight-tied logits** (`wte.weight` reused as output projection via `F.linear`)
- **Manual causal attention** in `CausalSelfAttention` (pedagogical); production swap is `F.scaled_dot_product_attention(is_causal=True)` — see guide 06 comments
- **No dropout** during pretraining (per GPT-2 paper §2)

---

## Test Architecture

| Tier | Path | Speed | Purpose |
|------|------|-------|---------|
| Unit | `tests/unit/` | < 1s/test | One module, isolated |
| Component | `tests/component/` | < 5s/test | Multi-module integration |
| System | `tests/system/` | ~30s | Full training pipeline |

System tests are marked `@pytest.mark.slow`. All stubs must be implemented
before any component or system tests pass.

Key fixtures in `tests/conftest.py`:
- `tiny_config` → `GPT2Config(vocab=64, ctx=16, embd=32, heads=4, layers=2)`
- `model` → fresh `GPT2Model(tiny_config)`, eval mode, seed 42
- `fake_ids` → shape `(2, 8)`, seed 0
- `random_3d_tensor` → shape `(2, 8, 32)`, seed 2

---

## Common Errors & Pitfalls

### 1. `NotImplementedError` in tests

**Symptom:** Tests fail with `NotImplementedError`.
**Cause:** The stub hasn't been implemented yet.
**Fix:** Work through guides in order: `guides/00_overview.md` → table.

---

### 2. Shape mismatch in `CausalSelfAttention`

**Symptom:** `RuntimeError: shape '...' is invalid for input of size ...`
**Cause:** Forgot to `.transpose(1, 2)` when reshaping to `(B, H, T, D)`.
**Fix:**
```python
q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,H,T,D)
```

---

### 3. Weight tying — `lm_head` accidentally created as `nn.Linear`

**Symptom:** `test_wte_and_lm_head_share_weights` fails.
**Cause:** Added `self.lm_head = nn.Linear(...)` in `__init__`.
**Fix:** Do NOT create an `nn.Linear`. Use `F.linear(h, self.transformer.wte.weight)` (or `h @ wte.weight.T`) in `forward`.

---

### 4. Cross-entropy loss not shifting (equal-length case)

**Symptom:** Loss is always near 0 at init (suspiciously perfect), or
`ValueError: Expected input batch_size (N) to match target batch_size (M)`.
**Cause:** When `logits` and `token_ids` are both shape `(B, T)`, you must shift
both by one position. Applying only one slice (or none) misaligns predictions
with targets.
**Fix (equal lengths — model forward / eval):**
```python
logits_flat  = logits[:, :-1, :].contiguous().view(-1, vocab_size)
targets_flat = token_ids[:, 1:].contiguous().view(-1)
```
**Fix (training batch — `token_ids` is `(B, T+1)`):** Feed `token_ids[:, :-1]`
to the model so logits are `(B, T, V)`; compare directly to `token_ids[:, 1:]`
with **no** extra logit shift. See pitfall 10.

---

### 5. Causal mask not applied (future tokens visible)

**Symptom:** `test_future_tokens_do_not_affect_past` fails.
**Cause:** Manual attention without a causal mask, or SDPA without `is_causal=True`.
**Fix (manual):** Apply a lower-triangular mask before softmax. **Fix (SDPA):** Add `is_causal=True` to the SDPA call.

---

### 6. Gradient accumulation bug

**Symptom:** Training loss increases after a few steps.
**Cause:** `optimizer.zero_grad()` called after `loss.backward()` instead of before `forward`.
**Fix:** The correct order is: `zero_grad → forward → loss → backward → clip → step`.

---

### 7. `LayerNorm` using biased variance

**Symptom:** Doesn't match `nn.LayerNorm` on the exact same inputs.
**Cause:** Using `x.var(unbiased=True)` (default) instead of `unbiased=False`.
**Fix:** `x.var(dim=-1, keepdim=True, unbiased=False)`.

---

### 8. Parameter-count test uses a loose magnitude bound

**Symptom:** `test_parameter_count_excludes_duplicate_lm_head` fails with
`assert 28032 < 14096`, even though weight tying is implemented correctly.
**Cause:** The assertion bounded `total` by `~2 * vocab_size * n_embd + 10_000`,
which only budgets for embedding-scale terms and ignores the transformer blocks
(they dominate, scaling as `n_layer * n_embd²`). It also can't distinguish a tied
head from an untied one (an untied `lm_head` is only `+vocab_size * n_embd`).
**Fix:** Derive the exact expected count from the architecture and assert
equality (`total == expected_tied`) plus `untied - tied == vocab_size * n_embd`.
See `_expected_param_count` in `tests/component/test_model_forward.py`.
**Prevention:** Assert exact, architecture-derived counts — never loose `<` bounds —
for structural invariants like parameter counts.

---

### 9. Static type error: `"Module" is not iterable`

**Symptom:** Pyright/Pylance flags `for layer in self.transformer["h"]`.
**Cause:** `nn.ModuleDict.__getitem__` is typed to return base `nn.Module` (no
`__iter__`), even though `"h"` holds an iterable `nn.ModuleList` at runtime.
**Fix:** `cast(nn.ModuleList, self.transformer["h"])` (zero runtime cost).
**Prevention:** When indexing `ModuleDict`/`ModuleList`, cast to the concrete
type if you rely on type-specific behaviour (iteration, indexing).

---

### 10. Training batch `(B, T+1)` passed directly to the model

**Symptom:** `AssertionError: Sequence too long: 17 > 16` during `Trainer.train`.
**Cause:** Data iterators yield `(B, n_ctx + 1)` tokens so the loss has a
next-token target for every input position. Passing the full batch to
`model(token_ids)` exceeds `n_ctx`.
**Fix:** Slice inputs before the forward pass; keep the full batch for loss:
```python
inputs = token_ids[:, :-1]          # (B, n_ctx)
logits = self.model(inputs)           # (B, n_ctx, vocab_size)
loss = cross_entropy_loss(logits, token_ids)  # token_ids still (B, n_ctx + 1)
```
**Prevention:** Treat `(B, T+1)` batches as "inputs + one extra target token",
not as a sequence length the model can consume.

---

### 11. LR schedule assert when `max_steps ≤ warmup_steps`

**Symptom:** `AssertionError: max_steps (10) must exceed warmup_steps (200)` on
the first training step in short smoke tests.
**Cause:** `TrainConfig` defaults to `warmup_steps=200`, but system tests use
small `max_steps` (10–50) without overriding warmup.
**Fix:** In `_update_lr`, clamp effective warmup before calling
`cosine_lr_schedule`:
```python
warmup_steps = min(config.warmup_steps, max(1, config.max_steps - 1))
```
**Prevention:** For any run where `max_steps` is small, either override
`warmup_steps` in `TrainConfig` or rely on the clamp in `_update_lr`.

---

## Key Technical Decisions

### Why manual attention in code, SDPA in production?

**Decision:** Implement `softmax(QK^T/√d_k) · V` manually in `CausalSelfAttention.forward`; document SDPA as the production one-liner.
**Context:** Manual attention materialises the `(B, H, T, T)` score matrix — O(T²) memory per layer. SDPA fuses scale → mask → softmax → matmul and can dispatch to FlashAttention.
**Alternative considered:** Ship SDPA from the start (as in nanochat). Faster and less memory, but hides the equation students are learning in guide 06.
**Rationale:** Clarity over performance for the learning path; comments show exactly how to swap in `F.scaled_dot_product_attention(q, k, v, is_causal=True)` once the math is understood.

---

### Why `frozen=True` on `GPT2Config`?

**Decision:** Make `GPT2Config` a frozen dataclass.
**Rationale:** Configs should not be mutated after creation. Accidental
`config.n_layer += 1` mid-training is a silent bug. Frozen makes the intent
explicit and prevents the bug entirely.

---

### Why no dropout?

**Decision:** No dropout in this implementation.
**Context:** The GPT-2 paper §2 uses no dropout for pretraining (stated explicitly).
Dropout is sometimes added for fine-tuning on small datasets.
**Alternative:** Add a `dropout` field to `GPT2Config` defaulting to 0.0 for forward-compatibility.

---

## Future Reimplementation Notes

### MLX (Apple Silicon)

- Replace `torch` with `mlx.core` and `mlx.nn`
- Same `GPT2Config` dataclass (no changes)
- Use `mlx.core.compile` instead of `torch.compile`
- Use MLX's lazy evaluation (no `.item()` inside training loop)
- Tests should import from `mlx_gpt2/` not `gpt2/`

### JAX + Flax

- Use `jax.numpy` everywhere `torch` is used
- Model: `flax.linen.Module` with `@nn.compact` decorators
- Training: `flax.training.train_state.TrainState`
- Differentiation: `jax.grad` or `jax.value_and_grad`
- Compilation: `jax.jit` on the train step
- Parallelism: `jax.vmap` for batching
- Tests should import from `jax_gpt2/` not `gpt2/`
