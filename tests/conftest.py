"""
Shared pytest fixtures for all test tiers.

Design principles:
  • All fixtures use the tiny config (vocab=64, ctx=16, embd=32, heads=4, layers=2)
    so tests run in milliseconds on CPU without any GPU.
  • Random seeds are fixed for determinism.
  • Fixtures are scoped at 'function' level by default so each test gets a
    fresh model/config (avoids state leaking between tests).

[best practice] Defining fixtures in conftest.py makes them available to all
tests in the directory without explicit imports — pytest discovers them automatically.
"""

import pytest
import torch

from gpt2.config import GPT2_TINY, GPT2Config
from gpt2.model import GPT2Model

# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_config() -> GPT2Config:
    """
    Minimal GPT2Config for fast CPU tests.

    Dimensions:
        vocab_size = 64
        n_ctx      = 16
        n_embd     = 32
        n_head     = 4
        n_layer    = 2
        head_dim   = 8
    """
    return GPT2_TINY


@pytest.fixture()
def small_batch_size() -> int:
    return 2


@pytest.fixture()
def seq_len(tiny_config: GPT2Config) -> int:
    """Default sequence length for tests — half the context window."""
    return tiny_config.n_ctx // 2  # 8


# ---------------------------------------------------------------------------
# Token tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_ids(tiny_config: GPT2Config, small_batch_size: int, seq_len: int) -> torch.Tensor:
    """
    Deterministic batch of token IDs.

    Shape: (batch_size, seq_len) = (2, 8)
    Values in [0, vocab_size) = [0, 64).
    """
    torch.manual_seed(0)
    return torch.randint(0, tiny_config.vocab_size, (small_batch_size, seq_len))


@pytest.fixture()
def fake_ids_for_loss(
    tiny_config: GPT2Config,
    small_batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Token IDs with T+1 tokens so cross_entropy_loss can form T (input, target) pairs.

    Shape: (batch_size, seq_len + 1) = (2, 9)
    """
    torch.manual_seed(0)
    return torch.randint(0, tiny_config.vocab_size, (small_batch_size, seq_len + 1))


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def model(tiny_config: GPT2Config) -> GPT2Model:
    """
    Fresh randomly-initialised GPT2Model using the tiny config.

    The model is on CPU and in eval mode by default. Tests that need
    train mode should call model.train() explicitly.
    """
    torch.manual_seed(42)
    m = GPT2Model(tiny_config)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Activation / tensor utilities
# ---------------------------------------------------------------------------


@pytest.fixture()
def random_float_tensor() -> torch.Tensor:
    """Shape (4, 8) float32 tensor, values drawn from N(0,1)."""
    torch.manual_seed(1)
    return torch.randn(4, 8)


@pytest.fixture()
def random_3d_tensor(tiny_config: GPT2Config, small_batch_size: int, seq_len: int) -> torch.Tensor:
    """Shape (B, T, C) = (2, 8, 32) float32 tensor matching tiny_config dims."""
    torch.manual_seed(2)
    return torch.randn(small_batch_size, seq_len, tiny_config.n_embd)
