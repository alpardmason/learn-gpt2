"""
Component tests for the generation utilities.

Tests cover top_k_filter, top_p_filter, and the generate loop.
Generation tests skip if the stubs are not yet implemented.
"""

import pytest
import torch

from gpt2.config import GPT2Config
from gpt2.generate import generate, top_k_filter, top_p_filter
from gpt2.model import GPT2Model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_stub(fn, *args, **kwargs):
    """Call fn; skip test if NotImplementedError is raised."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(str(e))


# ---------------------------------------------------------------------------
# top_k_filter
# ---------------------------------------------------------------------------

class TestTopKFilter:
    def test_keeps_top_k_values(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 0.5, 3.0]])
        result = _skip_if_stub(top_k_filter, logits, k=2)
        # Positions 0 and 2 (values 1.0 and 0.5) should be -inf
        assert result[0, 0].item() == float("-inf")
        assert result[0, 2].item() == float("-inf")
        # Positions 1 and 3 should survive
        assert result[0, 1].item() == 2.0
        assert result[0, 3].item() == 3.0

    def test_k_equals_vocab_size_keeps_all(self) -> None:
        logits = torch.randn(1, 16)
        result = _skip_if_stub(top_k_filter, logits, k=16)
        assert torch.isfinite(result).all()

    def test_k_equals_one_keeps_argmax(self) -> None:
        logits = torch.tensor([[0.1, 0.5, 0.9, 0.2]])
        result = _skip_if_stub(top_k_filter, logits, k=1)
        # Only the argmax (index 2) should survive
        assert result[0, 2].item() == pytest.approx(0.9)
        for i in [0, 1, 3]:
            assert result[0, i].item() == float("-inf")

    def test_output_shape_unchanged(self) -> None:
        logits = torch.randn(2, 32)
        result = _skip_if_stub(top_k_filter, logits, k=5)
        assert result.shape == logits.shape


# ---------------------------------------------------------------------------
# top_p_filter
# ---------------------------------------------------------------------------

class TestTopPFilter:
    def test_p_one_keeps_all(self) -> None:
        logits = torch.randn(1, 16)
        result = _skip_if_stub(top_p_filter, logits, p=1.0)
        # p=1.0 should keep all tokens
        assert torch.isfinite(result).all()

    def test_low_p_eliminates_most(self) -> None:
        # Strongly peaked distribution: first token dominates
        logits = torch.tensor([[10.0, 0.1, 0.1, 0.1, 0.1]])
        result = _skip_if_stub(top_p_filter, logits, p=0.9)
        # Most tokens should be filtered (set to -inf)
        neginf_count = (result == float("-inf")).sum().item()
        assert neginf_count >= 3

    def test_output_shape_unchanged(self) -> None:
        logits = torch.randn(2, 32)
        result = _skip_if_stub(top_p_filter, logits, p=0.9)
        assert result.shape == logits.shape


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_output_length(self, model: GPT2Model, tiny_config: GPT2Config) -> None:
        """generate returns (B, T + max_new_tokens)."""
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        out = _skip_if_stub(generate, model, idx, max_new_tokens=5)
        assert out.shape == (1, 9)

    def test_prompt_preserved(self, model: GPT2Model, tiny_config: GPT2Config) -> None:
        """The original prompt tokens must appear at the start of the output."""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 3))
        out = _skip_if_stub(generate, model, prompt, max_new_tokens=3)
        torch.testing.assert_close(out[:, :3], prompt)

    def test_temperature_zero_is_greedy(self, model: GPT2Model, tiny_config: GPT2Config) -> None:
        """temperature=0 (greedy) must be deterministic across two calls."""
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        out1 = _skip_if_stub(generate, model, idx, max_new_tokens=5, temperature=0.0)
        out2 = _skip_if_stub(generate, model, idx, max_new_tokens=5, temperature=0.0)
        torch.testing.assert_close(out1, out2)

    def test_same_seed_is_deterministic(self, model: GPT2Model, tiny_config: GPT2Config) -> None:
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        out1 = _skip_if_stub(generate, model, idx, max_new_tokens=5, seed=42)
        out2 = _skip_if_stub(generate, model, idx, max_new_tokens=5, seed=42)
        torch.testing.assert_close(out1, out2)

    def test_ids_in_valid_range(self, model: GPT2Model, tiny_config: GPT2Config) -> None:
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        out = _skip_if_stub(generate, model, idx, max_new_tokens=8)
        new_tokens = out[0, 4:]
        assert (new_tokens >= 0).all() and (new_tokens < tiny_config.vocab_size).all()
