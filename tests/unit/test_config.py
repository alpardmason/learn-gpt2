"""
Unit tests for GPT2Config.

Tests verify that the four published model sizes match the GPT-2 paper (Table 2)
and that the config validates its invariants.
"""

import pytest

from gpt2.config import (
    GPT2_LARGE,
    GPT2_MEDIUM,
    GPT2_SMALL,
    GPT2_TINY,
    GPT2_XL,
    GPT2Config,
)


class TestGPT2ConfigDefaults:
    def test_small_matches_paper(self) -> None:
        """GPT2_SMALL (117M) matches Table 2 of the GPT-2 paper."""
        assert GPT2_SMALL.vocab_size == 50257
        assert GPT2_SMALL.n_ctx == 1024
        assert GPT2_SMALL.n_embd == 768
        assert GPT2_SMALL.n_head == 12
        assert GPT2_SMALL.n_layer == 12

    def test_medium_matches_paper(self) -> None:
        assert GPT2_MEDIUM.n_embd == 1024
        assert GPT2_MEDIUM.n_head == 16
        assert GPT2_MEDIUM.n_layer == 24

    def test_large_matches_paper(self) -> None:
        assert GPT2_LARGE.n_embd == 1280
        assert GPT2_LARGE.n_head == 20
        assert GPT2_LARGE.n_layer == 36

    def test_xl_matches_paper(self) -> None:
        assert GPT2_XL.n_embd == 1600
        assert GPT2_XL.n_head == 25
        assert GPT2_XL.n_layer == 48

    def test_tiny_is_fast_for_tests(self) -> None:
        assert GPT2_TINY.vocab_size == 64
        assert GPT2_TINY.n_embd == 32
        assert GPT2_TINY.n_layer == 2


class TestGPT2ConfigValidation:
    def test_head_dim_property(self) -> None:
        cfg = GPT2Config(n_embd=64, n_head=8)
        assert cfg.head_dim == 8

    def test_invalid_head_divisor_raises(self) -> None:
        """n_embd must be divisible by n_head."""
        with pytest.raises(ValueError, match="divisible"):
            GPT2Config(n_embd=65, n_head=8)

    def test_frozen(self) -> None:
        """Config is immutable (frozen=True)."""
        cfg = GPT2Config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.n_layer = 99  # type: ignore[misc]

    def test_all_presets_have_valid_head_divisor(self) -> None:
        for cfg in [GPT2_SMALL, GPT2_MEDIUM, GPT2_LARGE, GPT2_XL, GPT2_TINY]:
            assert cfg.n_embd % cfg.n_head == 0, f"Failed for {cfg}"
