"""
Component tests for the Transformer Block.

A Block composes LayerNorm + CausalSelfAttention + LayerNorm + MLP with
pre-norm residual connections. These tests verify the composition works
correctly as a unit — going beyond the individual layer tests.
"""

import pytest
import torch

from gpt2.config import GPT2Config
from gpt2.model import Block


@pytest.fixture()
def block(tiny_config: GPT2Config) -> Block:
    torch.manual_seed(0)
    return Block(tiny_config)


class TestBlockShape:
    def test_shape_preserved(self, block: Block, tiny_config: GPT2Config) -> None:
        """Block output shape must equal input shape (B, T, C)."""
        x = torch.randn(2, 8, tiny_config.n_embd)
        y = block(x)
        assert y.shape == x.shape

    def test_single_token(self, block: Block, tiny_config: GPT2Config) -> None:
        x = torch.randn(1, 1, tiny_config.n_embd)
        y = block(x)
        assert y.shape == x.shape


class TestBlockResidual:
    def test_output_differs_from_input(self, block: Block, tiny_config: GPT2Config) -> None:
        """A non-trivial block should modify the residual stream."""
        torch.manual_seed(1)
        x = torch.randn(2, 8, tiny_config.n_embd)
        y = block(x)
        assert not torch.allclose(y, x), "Block output should differ from input"

    def test_residual_stream_not_zeroed(self, block: Block, tiny_config: GPT2Config) -> None:
        """Output must not be all zeros (residual would swamp zero sub-layers)."""
        x = torch.randn(2, 8, tiny_config.n_embd)
        y = block(x)
        assert y.abs().max().item() > 0.0


class TestBlockGradient:
    def test_gradient_flows_to_input(self, block: Block, tiny_config: GPT2Config) -> None:
        x = torch.randn(2, 8, tiny_config.n_embd, requires_grad=True)
        y = block(x).sum()
        y.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_all_parameters_receive_gradients(
        self, block: Block, tiny_config: GPT2Config
    ) -> None:
        """Every parameter in the block must have a gradient after backward."""
        x = torch.randn(2, 8, tiny_config.n_embd)
        block(x).sum().backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_no_nan_in_gradients(self, block: Block, tiny_config: GPT2Config) -> None:
        x = torch.randn(2, 8, tiny_config.n_embd)
        block(x).sum().backward()
        for name, param in block.named_parameters():
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
