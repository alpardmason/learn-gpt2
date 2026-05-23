# GPT-2 Reimplementation — Overview

Welcome. This project is a hardcore reimplementation of the original GPT-2 model
(Radford et al., 2019) in PyTorch. You have just read the paper. Now you will
internalize it by building it from scratch, one component at a time.

---

## How to Use This Project

**Workflow for each component:**

1. Open the component guide (e.g. `guides/04_gelu.md`)
2. Read the **Theory** section — understand the math before touching code
3. Do the **Warmup** exercise with pencil and paper
4. Open the corresponding source file and implement the stub
5. Run the specific test command to verify your implementation
6. Move to the next component

**Never skip the warmup.** The exercises are designed to surface exactly the
misconceptions that cause bugs in the implementation.

---

## Architecture Overview

```
Input token IDs: (B, T)
        │
        ▼
┌────────────────────────────────────────────────────┐
│  Token Embeddings   wte[token_ids]   (B, T, C)     │
│  + Position Embed.  wpe[0:T]         (B, T, C)     │
│                     h = tok + pos    (B, T, C)     │
└────────────────────────────────────────────────────┘
        │
        ▼  × N blocks
┌────────────────────────────────────────────────────┐
│  Block (repeated N = n_layer times)                │
│  ┌──────────────────────────────────────────────┐  │
│  │  x = x + CausalSelfAttention(LayerNorm1(x)) │  │
│  │  x = x + MLP(LayerNorm2(x))                 │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────┐
│  Final LayerNorm    ln_f(h)           (B, T, C)    │
│  Logits (weight-tied): h @ wte.weightᵀ (B,T,vocab) │
└────────────────────────────────────────────────────┘
        │
        ▼
Logits: (B, T, vocab_size)
```

Data dimensions:
- `B` = batch size
- `T` = sequence length (≤ n\_ctx = 1024)
- `C` = n\_embd (768 for GPT-2 Small)
- `H` = n\_head (12 for GPT-2 Small)
- `D` = head\_dim = C/H = 64 for GPT-2 Small

---

## Component Guide Index

Work through the guides in order. Each guide covers one task.

| # | Guide | Source File | Test Command |
|---|-------|------------|--------------|
| 01 | [Environment Setup](01_setup.md) | `pyproject.toml` | — |
| 02 | [Configuration](02_config.md) | `src/gpt2/config.py` | `uv run pytest tests/unit/test_config.py` |
| 03 | [Tokenizer](03_tokenizer.md) | `src/gpt2/tokenizer.py` | `uv run pytest tests/unit/test_tokenizer.py` |
| 04 | [GELU Activation](04_gelu.md) | `src/gpt2/model.py::GELU` | `uv run pytest tests/unit/test_gelu.py` |
| 05 | [Layer Normalization](05_layernorm.md) | `src/gpt2/model.py::LayerNorm` | `uv run pytest tests/unit/test_layernorm.py` |
| 06 | [Causal Self-Attention](06_causal_attention.md) | `src/gpt2/model.py::CausalSelfAttention` | `uv run pytest tests/unit/test_attention.py` |
| 07 | [MLP Block](07_mlp.md) | `src/gpt2/model.py::MLP` | `uv run pytest tests/unit/test_mlp.py` |
| 08 | [Transformer Block](08_block.md) | `src/gpt2/model.py::Block` | `uv run pytest tests/component/test_block.py` |
| 09 | [GPT-2 Forward Pass](09_model_forward.md) | `src/gpt2/model.py::GPT2Model` | `uv run pytest tests/component/test_model_forward.py` |
| 10 | [LM Loss](10_lm_loss.md) | `src/gpt2/train.py::cross_entropy_loss` | `uv run pytest tests/system/ -k loss` |
| 11 | [LR Schedule](11_lr_schedule.md) | `src/gpt2/train.py::cosine_lr_schedule` | `uv run pytest tests/system/ -k schedule` |
| 12 | [Training Loop](12_training_loop.md) | `src/gpt2/train.py::Trainer` | `uv run pytest tests/system/` |
| 13 | [Text Generation](13_generation.md) | `src/gpt2/generate.py` | `uv run pytest tests/component/test_generate.py` |
| 14 | [Further Reading](14_further_reading.md) | — | — |

---

## Model Size Reference (GPT-2 Paper, Table 2)

| Name | n\_layer | n\_embd | n\_head | Parameters |
|------|---------|--------|--------|------------|
| Small | 12 | 768 | 12 | ~117M |
| Medium | 24 | 1024 | 16 | ~345M |
| Large | 36 | 1280 | 20 | ~762M |
| XL | 48 | 1600 | 25 | ~1542M |
| **Tiny** (tests) | **2** | **32** | **4** | **~20K** |

---

## Quick Commands

```bash
# Install dependencies
uv sync --dev

# Run all fast tests (no GPU needed)
uv run pytest tests/unit/ tests/component/ -v

# Run full test suite including slow system tests
uv run pytest tests/ -v

# Run only tests for a specific component
uv run pytest tests/unit/test_gelu.py -v

# Lint the source code
uv run ruff check src/ tests/

# Train a tiny model (after implementing all stubs)
uv run python scripts/train.py --preset tiny --max_steps 200

# Generate text (after training)
uv run python scripts/generate.py --prompt "Once upon" --checkpoint checkpoints/gpt2_tiny_step200.pt
```
