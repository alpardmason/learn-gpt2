# Guide 01 — Environment Setup

## Theory

This project uses **uv** (a Rust-based Python package manager) to manage the
virtual environment and dependencies. It is faster than pip and reproducible
across machines.

The dependency stack:
- **PyTorch ≥ 2.6** — tensor operations and autograd
- **tiktoken** — OpenAI's fast BPE tokenizer (for Task 03)
- **pytest** — test runner (autograder)
- **ruff** — linter (enforces PEP 8 + import order + modern Python idioms)

## Warmup

Before running any code, answer these:

1. What does `uv sync --dev` do that `uv sync` alone does not?
2. Why do we pin `requires-python = ">=3.12"` in `pyproject.toml`?

## Your Task

No implementation needed. Verify the environment works:

```bash
# Create and activate the virtual environment
uv venv --python 3.12
uv sync --dev

# Confirm Python version
uv run python --version   # should print Python 3.12.x

# Confirm PyTorch is installed
uv run python -c "import torch; print(torch.__version__)"

# Run the config tests (all should PASS immediately — no stubs)
uv run pytest tests/unit/test_config.py -v
```

## Example Output

```
tests/unit/test_config.py::TestGPT2ConfigDefaults::test_small_matches_paper PASSED
tests/unit/test_config.py::TestGPT2ConfigDefaults::test_medium_matches_paper PASSED
...
5 passed in 0.12s
```

## Modern LLM Comparison

Modern ML projects (Hugging Face, Meta's LLaMA repo) use the same toolchain:
`pyproject.toml` + a fast package manager (uv, poetry, or pip-tools). The
key addition in production codebases is a CI pipeline (GitHub Actions) that
runs `ruff`, `mypy`, and `pytest` on every pull request.

## Run the Tests

```bash
uv run pytest tests/unit/test_config.py -v
```
