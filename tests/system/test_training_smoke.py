"""
System-level smoke tests for the full training pipeline.

These tests are marked @pytest.mark.slow because they run 50–100 training
steps (though still on a tiny model/CPU). They verify:
  1. Loss decreases from first step to last (the model learns)
  2. Gradient norms stay finite (no exploding gradients)
  3. The loss value at initialisation is near the theoretical expected value
  4. The Trainer + data loop integrates end-to-end without errors

Run with:
    uv run pytest tests/system/              # system only
    uv run pytest tests/ -m "not slow"       # skip these tests
    uv run pytest tests/                     # all tests including slow

Note: All stubs must be implemented before these tests can pass. They will
raise NotImplementedError if any component is incomplete.
"""

from __future__ import annotations

import pytest
import torch

from gpt2.config import GPT2_TINY, GPT2Config
from gpt2.model import GPT2Model
from gpt2.train import TrainConfig, Trainer, cosine_lr_schedule, cross_entropy_loss

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def repeating_sequence_iter(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    pattern_len: int = 4,
    seed: int = 42,
):
    """
    Yields batches of a short repeating pattern.

    Example: pattern [3, 7, 1, 5] → "3 7 1 5 3 7 1 5 3 7 1 5 …"

    A model CAN memorise this perfectly (loss → 0), so it is the minimal
    sanity check: loss MUST decrease significantly after ~50 steps.
    """
    torch.manual_seed(seed)
    pattern = torch.randint(0, vocab_size, (pattern_len,))
    # Tile the pattern to create a full sequence
    reps = (seq_len + pattern_len) // pattern_len + 1
    full = pattern.repeat(reps)[:seq_len + 1]  # +1 for loss shift
    while True:
        batch = full.unsqueeze(0).expand(batch_size, -1).contiguous()
        yield batch


def random_iter(vocab_size: int, seq_len: int, batch_size: int, seed: int = 42):
    """Yields random token batches (loss should not decrease below log(vocab_size))."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    while True:
        yield torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=rng)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestCrossEntropyLoss:
    def test_initial_loss_near_log_vocab(self) -> None:
        """
        At random initialisation the model assigns roughly uniform probabilities,
        so the expected loss is log(vocab_size).

        [theory] If the model outputs uniform logits, softmax gives p = 1/V for
        every token. Cross-entropy = -log(1/V) = log(V).
        For vocab_size=64: expected loss ≈ log(64) ≈ 4.16.
        """
        torch.manual_seed(0)
        cfg = GPT2_TINY
        model = GPT2Model(cfg)
        model.eval()

        idx = torch.randint(0, cfg.vocab_size, (4, cfg.n_ctx))
        with torch.no_grad():
            logits = model(idx)
        loss = cross_entropy_loss(logits, idx)

        import math
        expected = math.log(cfg.vocab_size)
        # Allow ±1.5 nats from the theoretical value
        assert abs(loss.item() - expected) < 1.5, (
            f"Initial loss {loss.item():.3f} is far from log(vocab_size)={expected:.3f}"
        )

    def test_loss_is_scalar(self, tiny_config: GPT2Config) -> None:
        torch.manual_seed(0)
        model = GPT2Model(tiny_config)
        idx = torch.randint(0, tiny_config.vocab_size, (2, 8))
        with torch.no_grad():
            logits = model(idx)
        loss = cross_entropy_loss(logits, idx)
        assert loss.shape == torch.Size([])

    def test_perfect_prediction_gives_low_loss(self, tiny_config: GPT2Config) -> None:
        """If logits are very confident and correct, loss should be near 0."""
        B, T, V = 2, 4, tiny_config.vocab_size
        # Build token IDs and perfect logits
        ids = torch.zeros(B, T + 1, dtype=torch.long)
        ids[:, :] = 3  # all tokens are 3 → target is also 3
        logits = torch.full((B, T, V), -1e9)
        logits[:, :, 3] = 1e9  # extremely confident: token 3 always predicted
        loss = cross_entropy_loss(logits, ids)
        assert loss.item() < 0.01


# ---------------------------------------------------------------------------
# LR Schedule tests
# ---------------------------------------------------------------------------

class TestCosineSchedule:
    def test_starts_at_zero(self) -> None:
        lr = cosine_lr_schedule(0, max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000)
        assert lr == pytest.approx(0.0, abs=1e-9)

    def test_peaks_at_end_of_warmup(self) -> None:
        lr = cosine_lr_schedule(100, max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000)
        assert lr == pytest.approx(1e-3, rel=1e-3)

    def test_ends_at_min_lr(self) -> None:
        lr = cosine_lr_schedule(1000, max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000)
        assert lr == pytest.approx(1e-4, rel=1e-3)

    def test_monotone_during_warmup(self) -> None:
        lrs = [
            cosine_lr_schedule(t, max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000)
            for t in range(0, 101)
        ]
        assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_monotone_decreasing_after_warmup(self) -> None:
        lrs = [
            cosine_lr_schedule(t, max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000)
            for t in range(100, 1001, 50)
        ]
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_clips_at_min_after_max_steps(self) -> None:
        lr = cosine_lr_schedule(9999, max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000)
        assert lr == pytest.approx(1e-4, rel=1e-3)


# ---------------------------------------------------------------------------
# Full training loop smoke test
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrainingSmokeTest:
    def test_loss_decreases_on_repeating_sequence(self) -> None:
        """
        A model trained on a repeating pattern for 50 steps must have a lower
        loss at the end than at the beginning (the model is learning).

        This is the primary integration test: it exercises every component
        end-to-end (model forward, loss, backward, optimizer, LR schedule).
        """
        cfg = GPT2_TINY
        torch.manual_seed(0)
        model = GPT2Model(cfg)
        train_cfg = TrainConfig(
            max_steps=50,
            batch_size=4,
            learning_rate=1e-3,
            warmup_steps=5,
            device="cpu",
        )
        trainer = Trainer(model, train_cfg)

        data = repeating_sequence_iter(
            vocab_size=cfg.vocab_size,
            seq_len=cfg.n_ctx,
            batch_size=train_cfg.batch_size,
        )

        losses = []
        for step, loss in trainer.train(data):
            losses.append(loss)

        assert len(losses) == 50

        initial_loss = losses[0]
        final_loss = losses[-1]
        assert final_loss < initial_loss * 0.9, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_gradient_norms_stay_finite(self) -> None:
        """No gradient explosion: norms must be finite at every step."""
        cfg = GPT2_TINY
        torch.manual_seed(1)
        model = GPT2Model(cfg)
        train_cfg = TrainConfig(max_steps=20, batch_size=2, learning_rate=1e-3, device="cpu")
        trainer = Trainer(model, train_cfg)

        data = random_iter(vocab_size=cfg.vocab_size, seq_len=cfg.n_ctx, batch_size=2)

        for step, loss in trainer.train(data):
            total_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ) ** 0.5
            assert torch.isfinite(torch.tensor(total_norm)), (
                f"Infinite gradient norm at step {step}"
            )
            assert torch.isfinite(torch.tensor(loss)), (
                f"Infinite loss at step {step}"
            )

    def test_loss_is_not_nan(self) -> None:
        cfg = GPT2_TINY
        torch.manual_seed(2)
        model = GPT2Model(cfg)
        train_cfg = TrainConfig(max_steps=10, batch_size=2, device="cpu")
        trainer = Trainer(model, train_cfg)

        data = random_iter(vocab_size=cfg.vocab_size, seq_len=cfg.n_ctx, batch_size=2)
        for _, loss in trainer.train(data):
            assert not torch.isnan(torch.tensor(loss)), "Loss is NaN"
