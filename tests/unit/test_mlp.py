"""
Unit tests for the MLP (feed-forward network).

Tests verify:
  1. Output shape matches input (B, T, C) — dimension is preserved end-to-end
  2. Internal expansion: c_fc output has dim 4*n_embd
  3. Weight shapes are correct
  4. Gradient flows through both layers and the GELU activation
"""

import pytest
import torch

from gpt2.config import GPT2Config
from gpt2.model import MLP


@pytest.fixture()
def mlp(tiny_config: GPT2Config) -> MLP:
    torch.manual_seed(0)
    return MLP(tiny_config)


class TestMLPShape:
    def test_output_shape_3d(self, mlp: MLP, tiny_config: GPT2Config) -> None:
        """MLP output must be (B, T, C) for any (B, T, C) input."""
        x = torch.randn(2, 8, tiny_config.n_embd)
        y = mlp(x)
        assert y.shape == (2, 8, tiny_config.n_embd)

    def test_output_shape_2d(self, mlp: MLP, tiny_config: GPT2Config) -> None:
        """Works with (T, C) input too (no batch dim)."""
        x = torch.randn(8, tiny_config.n_embd)
        y = mlp(x)
        assert y.shape == (8, tiny_config.n_embd)


class TestMLPWeightShapes:
    def test_c_fc_expands_4x(self, mlp: MLP, tiny_config: GPT2Config) -> None:
        """c_fc maps n_embd → 4*n_embd."""
        assert mlp.c_fc.weight.shape == (4 * tiny_config.n_embd, tiny_config.n_embd)

    def test_c_proj_contracts(self, mlp: MLP, tiny_config: GPT2Config) -> None:
        """c_proj maps 4*n_embd → n_embd."""
        assert mlp.c_proj.weight.shape == (tiny_config.n_embd, 4 * tiny_config.n_embd)

    def test_bias_shapes(self, mlp: MLP, tiny_config: GPT2Config) -> None:
        assert mlp.c_fc.bias.shape == (4 * tiny_config.n_embd,)
        assert mlp.c_proj.bias.shape == (tiny_config.n_embd,)


class TestMLPGradient:
    def test_gradient_through_mlp(self, mlp: MLP, tiny_config: GPT2Config) -> None:
        x = torch.randn(2, 8, tiny_config.n_embd, requires_grad=True)
        y = mlp(x).sum()
        y.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_weight_gradients(self, mlp: MLP, tiny_config: GPT2Config) -> None:
        """Both linear layers must receive gradients."""
        x = torch.randn(2, 8, tiny_config.n_embd)
        mlp(x).sum().backward()
        assert mlp.c_fc.weight.grad is not None
        assert mlp.c_proj.weight.grad is not None
