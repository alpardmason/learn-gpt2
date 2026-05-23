"""
Unit tests for the GELU activation function.

Tests verify:
  1. Known numerical values (from the GELU paper)
  2. Output shape matches input shape
  3. Agreement with PyTorch's built-in GELU(approximate='tanh')
  4. Gradient flows through the function (autograd works)
"""


import pytest
import torch
import torch.nn as nn

from gpt2.model import GELU


@pytest.fixture()
def gelu() -> GELU:
    return GELU()


class TestGELUValues:
    def test_zero_input(self, gelu: GELU) -> None:
        """GELU(0) = 0 exactly (the function passes through the origin)."""
        x = torch.tensor(0.0)
        assert gelu(x).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_half(self, gelu: GELU) -> None:
        """GELU(0.5) ≈ 0.3457 (reference value from GELU paper)."""
        x = torch.tensor(0.5)
        assert gelu(x).item() == pytest.approx(0.3457, abs=1e-3)

    def test_one(self, gelu: GELU) -> None:
        """GELU(1.0) ≈ 0.8413."""
        x = torch.tensor(1.0)
        assert gelu(x).item() == pytest.approx(0.8413, abs=1e-3)

    def test_negative_large(self, gelu: GELU) -> None:
        """GELU(x) → 0 as x → −∞ (unlike ReLU which equals 0 for x < 0)."""
        x = torch.tensor(-10.0)
        assert gelu(x).item() == pytest.approx(0.0, abs=1e-4)

    def test_positive_large(self, gelu: GELU) -> None:
        """GELU(x) ≈ x for large positive x (converges to identity)."""
        x = torch.tensor(10.0)
        assert gelu(x).item() == pytest.approx(10.0, abs=1e-3)


class TestGELUShape:
    def test_scalar(self, gelu: GELU) -> None:
        x = torch.tensor(1.0)
        assert gelu(x).shape == x.shape

    def test_1d(self, gelu: GELU) -> None:
        x = torch.randn(16)
        assert gelu(x).shape == (16,)

    def test_3d(self, gelu: GELU) -> None:
        x = torch.randn(2, 8, 32)
        assert gelu(x).shape == (2, 8, 32)


class TestGELUAgreement:
    def test_matches_torch_gelu_tanh(self, gelu: GELU) -> None:
        """Our GELU must match nn.GELU(approximate='tanh') on random inputs."""
        torch.manual_seed(0)
        x = torch.randn(4, 8, 32)
        ref = nn.GELU(approximate="tanh")
        torch.testing.assert_close(gelu(x), ref(x), atol=1e-5, rtol=1e-5)


class TestGELUGradient:
    def test_gradient_exists(self, gelu: GELU) -> None:
        """Autograd must compute gradients through GELU."""
        x = torch.randn(4, requires_grad=True)
        y = gelu(x).sum()
        y.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
