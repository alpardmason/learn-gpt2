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
- **Weight-tied logits** (`wte.weight` reused as output projection)
- **Causal mask via `F.scaled_dot_product_attention(is_causal=True)`**
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
**Fix:** Do NOT create an `nn.Linear`. Use `h @ self.transformer.wte.weight.T` in `forward`.

---

### 4. Cross-entropy loss not shifting

**Symptom:** Loss is always near 0 at init (suspiciously perfect).
**Cause:** Passing `logits` and `token_ids` without the `[:, :-1]` / `[:, 1:]` shift.
**Fix:**
```python
logits_shifted  = logits[:, :-1, :].contiguous()
targets_shifted = token_ids[:, 1:].contiguous()
```

---

### 5. Causal mask not applied (future tokens visible)

**Symptom:** `test_future_tokens_do_not_affect_past` fails.
**Cause:** Using `F.scaled_dot_product_attention(q, k, v)` without `is_causal=True`.
**Fix:** Add `is_causal=True` to the SDPA call.

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

## Key Technical Decisions

### Why `F.scaled_dot_product_attention` instead of manual attention?

**Decision:** Use PyTorch ≥ 2.0's built-in SDPA.
**Context:** Manual attention requires materialising the `(B, H, T, T)` attention
weight matrix, which is O(T²) memory. SDPA auto-dispatches to FlashAttention
when hardware supports it, falling back to math attention otherwise.
**Alternative considered:** Manually implement `softmax(QK^T/√d_k) · V` as in
the OpenAI reference code. This is didactically clearer but O(T²) in memory.
**Rationale:** Clarity is preserved (the equation is in the guide); correctness
is guaranteed by PyTorch; students see a production-quality pattern.

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
