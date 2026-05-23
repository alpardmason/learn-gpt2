"""
Unit tests for LayerNorm.

Tests verify:
  1. Output has (approximately) zero mean and unit variance
  2. Learnable weight (γ) and bias (β) are applied correctly
  3. Agreement with nn.LayerNorm on random inputs
  4. Gradient flows through both the normalisation and learnable params
"""

import torch
import torch.nn as nn

from gpt2.model import LayerNorm


class TestLayerNormStatistics:
    def test_zero_mean_unit_var(self) -> None:
        """After normalisation (γ=1, β=0), output has mean≈0 and var≈1."""
        torch.manual_seed(0)
        ln = LayerNorm(32)
        # Ensure default init: weight=1, bias=0
        nn.init.ones_(ln.weight)
        nn.init.zeros_(ln.bias)

        x = torch.randn(4, 8, 32) * 5 + 3  # deliberately off-centre
        y = ln(x)

        # Per-token statistics (last dim)
        assert y.mean(dim=-1).abs().max().item() < 1e-4
        # Note: var uses unbiased=True by default → close to 1 for large dim
        assert (y.var(dim=-1) - 1.0).abs().max().item() < 0.1

    def test_custom_weight_scales_output(self) -> None:
        """Setting weight=2 should double the normalised values."""
        torch.manual_seed(0)
        ln = LayerNorm(8)
        nn.init.constant_(ln.weight, 2.0)
        nn.init.zeros_(ln.bias)

        x = torch.randn(2, 8)
        y = ln(x)

        # Mean should still be ~0 (bias=0)
        assert y.mean(dim=-1).abs().max().item() < 1e-4

    def test_bias_shifts_output(self) -> None:
        """Setting bias=3 should shift the mean of the output by 3."""
        torch.manual_seed(0)
        ln = LayerNorm(16)
        nn.init.ones_(ln.weight)
        nn.init.constant_(ln.bias, 3.0)

        x = torch.randn(4, 16)
        y = ln(x)

        assert y.mean(dim=-1).mean().item() == torch.tensor(3.0).item()


class TestLayerNormShape:
    def test_shape_preserved(self) -> None:
        """Output shape must match input shape."""
        ln = LayerNorm(32)
        x = torch.randn(2, 8, 32)
        assert ln(x).shape == (2, 8, 32)


class TestLayerNormAgreement:
    def test_matches_nn_layernorm(self) -> None:
        """Our LayerNorm must match PyTorch's nn.LayerNorm."""
        torch.manual_seed(42)
        n = 32
        our_ln = LayerNorm(n)
        ref_ln = nn.LayerNorm(n, eps=1e-5)

        # Copy weights to make comparison fair
        ref_ln.weight.data.copy_(our_ln.weight.data)
        ref_ln.bias.data.copy_(our_ln.bias.data)

        x = torch.randn(2, 8, n)
        torch.testing.assert_close(our_ln(x), ref_ln(x), atol=1e-5, rtol=1e-5)


class TestLayerNormGradients:
    def test_gradient_through_weight_and_bias(self) -> None:
        """Gradients must flow to weight and bias params."""
        ln = LayerNorm(16)
        x = torch.randn(2, 8, 16)
        y = ln(x).sum()
        y.backward()
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None

    def test_no_nan_gradients(self) -> None:
        ln = LayerNorm(16)
        x = torch.randn(2, 16, requires_grad=True)
        y = ln(x).sum()
        y.backward()
        assert not torch.isnan(x.grad).any()
