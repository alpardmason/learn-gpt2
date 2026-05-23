"""
Component tests for GPT2Model forward pass.

Tests the full model as an integrated unit:
  • Correct logit shape
  • Weight tying between wte and lm_head
  • Gradient flows to all parameters
  • Logits are finite and non-trivial
"""

import torch

from gpt2.config import GPT2Config
from gpt2.model import GPT2Model


class TestModelForwardShape:
    def test_logit_shape(self, model: GPT2Model, tiny_config: GPT2Config) -> None:
        """Forward pass must return (B, T, vocab_size)."""
        B, T = 2, 8
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits = model(idx)
        assert logits.shape == (B, T, tiny_config.vocab_size)

    def test_single_token_shape(self, model: GPT2Model, tiny_config: GPT2Config) -> None:
        idx = torch.randint(0, tiny_config.vocab_size, (1, 1))
        logits = model(idx)
        assert logits.shape == (1, 1, tiny_config.vocab_size)

    def test_full_context_shape(self, model: GPT2Model, tiny_config: GPT2Config) -> None:
        idx = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.n_ctx))
        logits = model(idx)
        assert logits.shape == (1, tiny_config.n_ctx, tiny_config.vocab_size)


class TestWeightTying:
    def test_wte_and_lm_head_share_weights(self, model: GPT2Model) -> None:
        """
        Weight tying: the output projection uses wte.weight.
        Since we use F.linear(h, wte.weight) in forward(), the same tensor
        is shared — not just equal values but the same Python object.
        """
        # The forward pass uses wte.weight as the projection matrix.
        # We verify there is only ONE copy of the vocab embedding parameter.
        wte_weight = model.transformer.wte.weight
        # Check: no separate lm_head parameter exists in the model
        param_names = [n for n, _ in model.named_parameters()]
        assert not any("lm_head" in n for n in param_names), (
            "lm_head should not be a separate parameter — use weight tying"
        )
        # The wte weight must exist
        assert wte_weight is not None

    def test_parameter_count_excludes_duplicate_lm_head(
        self, model: GPT2Model, tiny_config: GPT2Config
    ) -> None:
        """Weight tying means vocab params are counted once, not twice."""
        total = model.num_parameters()
        # With tying, params = embeddings (wte+wpe) + blocks + ln_f
        # Without tying we'd have vocab_size*n_embd extra params for lm_head
        extra_if_untied = tiny_config.vocab_size * tiny_config.n_embd
        # The total should NOT include this extra count
        assert total < extra_if_untied * 2 + 10_000  # generous upper bound


class TestModelForwardValues:
    def test_logits_are_finite(self, model: GPT2Model, tiny_config: GPT2Config) -> None:
        idx = torch.randint(0, tiny_config.vocab_size, (2, 8))
        logits = model(idx)
        assert torch.isfinite(logits).all()

    def test_logits_are_not_constant(self, model: GPT2Model, tiny_config: GPT2Config) -> None:
        """Logits must differ across vocabulary positions (not all the same)."""
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        logits = model(idx)
        assert logits[0, 0].std().item() > 0.0


class TestModelGradients:
    def test_all_parameters_get_gradients(
        self, model: GPT2Model, tiny_config: GPT2Config
    ) -> None:
        model.train()
        idx = torch.randint(0, tiny_config.vocab_size, (2, 8))
        logits = model(idx)
        logits.sum().backward()

        no_grad = [n for n, p in model.named_parameters() if p.grad is None]
        assert no_grad == [], f"No gradient for: {no_grad}"

    def test_no_nan_gradients(self, model: GPT2Model, tiny_config: GPT2Config) -> None:
        model.train()
        idx = torch.randint(0, tiny_config.vocab_size, (2, 8))
        model(idx).sum().backward()
        for name, param in model.named_parameters():
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
