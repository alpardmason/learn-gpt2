"""
Unit tests for CausalSelfAttention.

Tests verify:
  1. Output shape matches input shape (B, T, C)
  2. Causal masking: future tokens cannot influence past positions
  3. Weight shapes match the config
  4. Gradient flows through the attention module
"""

import pytest
import torch

from gpt2.config import GPT2Config
from gpt2.model import CausalSelfAttention


@pytest.fixture()
def attn(tiny_config: GPT2Config) -> CausalSelfAttention:
    torch.manual_seed(0)
    return CausalSelfAttention(tiny_config)


class TestAttentionShape:
    def test_output_shape(self, attn: CausalSelfAttention, tiny_config: GPT2Config) -> None:
        """Output must be (B, T, C) — same shape as input."""
        B, T, C = 2, 8, tiny_config.n_embd
        x = torch.randn(B, T, C)
        y = attn(x)
        assert y.shape == (B, T, C)

    def test_single_token(self, attn: CausalSelfAttention, tiny_config: GPT2Config) -> None:
        """A single-token sequence should work (T=1)."""
        x = torch.randn(1, 1, tiny_config.n_embd)
        y = attn(x)
        assert y.shape == (1, 1, tiny_config.n_embd)

    def test_full_context(self, attn: CausalSelfAttention, tiny_config: GPT2Config) -> None:
        """A full-length sequence (T = n_ctx) should work."""
        x = torch.randn(1, tiny_config.n_ctx, tiny_config.n_embd)
        y = attn(x)
        assert y.shape == (1, tiny_config.n_ctx, tiny_config.n_embd)


class TestCausalMask:
    def test_future_tokens_do_not_affect_past(
        self, attn: CausalSelfAttention, tiny_config: GPT2Config
    ) -> None:
        """
        The output at position t must not depend on inputs at positions > t.
        We verify this by comparing outputs with and without a different future token.
        """
        torch.manual_seed(1)
        B, T, C = 1, 6, tiny_config.n_embd
        x1 = torch.randn(B, T, C)
        x2 = x1.clone()
        # Perturb the last position
        x2[:, -1, :] = torch.randn(B, C)

        y1 = attn(x1)
        y2 = attn(x2)

        # Positions 0..T-2 must be identical (not influenced by position T-1)
        torch.testing.assert_close(y1[:, :-1, :], y2[:, :-1, :])
        # Position T-1 may differ
        assert not torch.allclose(y1[:, -1, :], y2[:, -1, :])


class TestAttentionWeights:
    def test_c_attn_shape(self, attn: CausalSelfAttention, tiny_config: GPT2Config) -> None:
        """c_attn projects n_embd → 3*n_embd for Q, K, V."""
        assert attn.c_attn.weight.shape == (3 * tiny_config.n_embd, tiny_config.n_embd)

    def test_c_proj_shape(self, attn: CausalSelfAttention, tiny_config: GPT2Config) -> None:
        """c_proj maps n_embd → n_embd."""
        assert attn.c_proj.weight.shape == (tiny_config.n_embd, tiny_config.n_embd)

    def test_head_dim(self, attn: CausalSelfAttention, tiny_config: GPT2Config) -> None:
        assert attn.head_dim == tiny_config.n_embd // tiny_config.n_head


class TestAttentionGradient:
    def test_gradient_exists(self, attn: CausalSelfAttention, tiny_config: GPT2Config) -> None:
        x = torch.randn(2, 8, tiny_config.n_embd, requires_grad=True)
        y = attn(x).sum()
        y.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
