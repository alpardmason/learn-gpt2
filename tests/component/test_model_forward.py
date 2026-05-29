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


def _expected_param_count(cfg: GPT2Config, *, tied: bool) -> int:
    """Analytically derive the exact trainable parameter count.

    Mirrors the architecture in ``model.py`` so the test fails loudly if the
    structure ever changes:
      • LayerNorm: weight + bias = 2 * n_embd
      • nn.Linear(in, out): in * out + out (bias on by default)
      • MLP hidden width: 4 * n_embd
    Setting ``tied=False`` adds a separate ``vocab_size * n_embd`` lm_head.
    """
    e = cfg.n_embd

    def linear(in_f: int, out_f: int) -> int:
        return in_f * out_f + out_f

    layer_norm = 2 * e

    per_block = (
        layer_norm  # ln_1
        + linear(e, 3 * e)  # c_attn (Q, K, V fused)
        + linear(e, e)  # attention output proj
        + layer_norm  # ln_2
        + linear(e, 4 * e)  # mlp c_fc
        + linear(4 * e, e)  # mlp c_proj
    )

    embeddings = cfg.vocab_size * e + cfg.n_ctx * e  # wte + wpe
    total = embeddings + cfg.n_layer * per_block + layer_norm  # + ln_f
    if not tied:
        total += cfg.vocab_size * e  # separate lm_head weight
    return total


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
        """Weight tying means the vocab projection is counted once, not twice.

        Rather than a loose magnitude bound (which can't distinguish a tied head
        from an untied one), we reconstruct the expected parameter count from the
        architecture and assert an exact match. An untied model would have exactly
        ``vocab_size * n_embd`` extra parameters for a separate lm_head.
        """
        total = model.num_parameters()
        expected_tied = _expected_param_count(tiny_config, tied=True)
        untied = _expected_param_count(tiny_config, tied=False)

        # The whole point of weight tying: total equals the tied count exactly,
        # i.e. it excludes the duplicate lm_head projection an untied model adds.
        assert total == expected_tied
        assert untied - expected_tied == tiny_config.vocab_size * tiny_config.n_embd


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
