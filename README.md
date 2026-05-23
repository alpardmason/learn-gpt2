# learn-gpt2

An educational reimplementation of the original GPT-2 (Radford et al., 2019)
using PyTorch and modern Python 3.12+, structured as a hands-on assignment in
the style of Stanford CS336.

---

## What This Is

This repository is a **skeleton** — the architecture, tests, and guides are
provided, but the core implementation functions are left as `raise NotImplementedError`
stubs for you to fill in. You work through the components bottom-up, verifying
each one with the autograder before moving to the next.

---

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | **Active development** — ongoing GPT-2 reimplementation work lives here |
| `skeleton` | **Frozen student starting point** — stubs, guides, and autograder tests (unchanged) |

**On `main`:** clone and work as usual — this is where implementation progress is tracked.

**To reimplement from scratch** (same starting point every time), use the frozen `skeleton` branch:

```bash
git clone git@github.com:alpardmason/learn-gpt2.git
cd learn-gpt2
git checkout skeleton          # frozen stubs — do not commit here
# or branch off it:
git checkout -b my-gpt2 skeleton
uv venv --python 3.12
uv sync --dev
```

Use `guides/00_overview.md` as the task index. Stubs raise `NotImplementedError` until you implement them.

---

**You will implement:**
- GELU activation (tanh approximation)
- Layer Normalization
- Causal Multi-Head Self-Attention
- Position-wise MLP (Feed-Forward Network)
- Transformer Block (pre-norm residual)
- Full GPT-2 Forward Pass (weight-tied logits)
- Next-token Cross-Entropy Loss
- Cosine LR Schedule with warmup
- Training Loop (AdamW + gradient clipping)
- Top-k / Top-p Autoregressive Generation
- BPE Tokenizer (thin wrapper around tiktoken)

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- The GPT-2 paper (provided in `references/`)
- Basic PyTorch familiarity (tensors, autograd, `nn.Module`)

---

## Getting Started

```bash
# 1. Clone the repository
git clone <repo-url>
cd learn-gpt2

# 2. Create virtual environment and install dependencies
uv venv --python 3.12
uv sync --dev

# 3. Verify setup — these should pass immediately (no stubs required)
uv run pytest tests/unit/test_config.py -v
uv run pytest tests/unit/test_tokenizer.py -k "not bpe" -v
```

---

## Project Structure

```
learn-gpt2/
├── guides/                 ← Per-component student guides (start here)
│   ├── 00_overview.md      ← Architecture diagram, task index, quick commands
│   ├── 01_setup.md
│   ├── 02_config.md
│   ├── ...
│   └── 14_further_reading.md
├── src/
│   └── gpt2/
│       ├── config.py       ← GPT2Config dataclass + model size presets
│       ├── tokenizer.py    ← CharTokenizer (complete) + BPETokenizer (stub)
│       ├── model.py        ← GELU, LayerNorm, Attention, MLP, Block, GPT2Model
│       ├── train.py        ← cross_entropy_loss, cosine_lr_schedule, Trainer
│       └── generate.py     ← top_k_filter, top_p_filter, generate
├── tests/
│   ├── conftest.py         ← Shared fixtures (tiny_config, fake_ids, ...)
│   ├── unit/               ← Component-level autograder (fast, CPU)
│   ├── component/          ← Integration tests (block, model, generation)
│   └── system/             ← End-to-end training smoke tests (slow)
├── scripts/
│   ├── train.py            ← CLI: train a GPT-2 model
│   └── generate.py         ← CLI: generate text from a trained checkpoint
├── references/             ← Local only (gitignored); see references/INDEX.md
│   ├── INDEX.md            ← Local catalog (not versioned)
│   ├── language_models_are_unsupervised_multitask_learners.pdf
│   ├── gelu_paper.pdf
│   ├── gpt-2/              ← Original OpenAI TensorFlow code (reference)
│   └── nanochat/           ← Modern PyTorch reference (Karpathy)
├── .cursor/                ← Local Cursor rules (gitignored); see .cursor/INDEX.md
└── pyproject.toml
```

---

## Recommended Workflow

Open `guides/00_overview.md` and follow the component index in order:

| Step | Guide | Tests |
|------|-------|-------|
| 1 | `guides/01_setup.md` | Verify environment |
| 2 | `guides/02_config.md` | `test_config.py` |
| 3 | `guides/03_tokenizer.md` | `test_tokenizer.py` |
| 4 | `guides/04_gelu.md` | `test_gelu.py` |
| 5 | `guides/05_layernorm.md` | `test_layernorm.py` |
| 6 | `guides/06_causal_attention.md` | `test_attention.py` |
| 7 | `guides/07_mlp.md` | `test_mlp.py` |
| 8 | `guides/08_block.md` | `test_block.py` |
| 9 | `guides/09_model_forward.md` | `test_model_forward.py` |
| 10 | `guides/10_lm_loss.md` | `test_training_smoke.py -k loss` |
| 11 | `guides/11_lr_schedule.md` | `test_training_smoke.py -k schedule` |
| 12 | `guides/12_training_loop.md` | `test_training_smoke.py` |
| 13 | `guides/13_generation.md` | `test_generate.py` |

**Do not skip the warmup exercises.** They surface the misconceptions that cause bugs.

---

## Running Tests

```bash
# Fast autograder — unit tests only (no GPU, < 30 seconds)
uv run pytest tests/unit/ -v

# Component (integration) tests
uv run pytest tests/component/ -v

# All except the slow system tests
uv run pytest tests/ -m "not slow" -v

# Full suite including training smoke test
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/
```

---

## After Completing All Tasks

```bash
# Train a tiny model on synthetic data
uv run python scripts/train.py --preset tiny --max_steps 500

# Generate text (will need a checkpoint)
uv run python scripts/generate.py \
    --checkpoint checkpoints/gpt2_tiny_step500.pt \
    --prompt "Once upon a time" \
    --max_new_tokens 50 --top_k 40
```

---

## Reference Priority

When the GPT-2 paper lacks detail, consult in this order (local copies in
`references/` — not versioned; see [`references/INDEX.md`](references/INDEX.md)
for setup):

1. **GPT-2 paper** (`references/language_models_are_unsupervised_multitask_learners.pdf`)
2. **OpenAI TensorFlow code** (`references/gpt-2/src/model.py`) — implementation specifics
3. **nanochat** (`references/nanochat/`) — modern PyTorch patterns only

---

## Comment Conventions in Source Code

| Tag | Meaning |
|-----|---------|
| `[theory]` | Mathematical concept from the paper |
| `[best practice]` | Industry practice for career preparation |
| `[modern]` | How modern LLMs (LLaMA, GPT-4) differ |
| `[nanochat]` | Where a PyTorch pattern references nanochat |

---

## Future Reimplementations (Planned)

- **MLX** (Apple Silicon): same architecture using `mlx` and `mlx.nn`
- **JAX + Flax**: functional style with `jax.grad`, `jax.jit`, `flax.linen`

---

## License

MIT
