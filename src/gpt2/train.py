"""
GPT-2 training utilities.

Components:
  cross_entropy_loss    — next-token prediction loss (Task 10)
  cosine_lr_schedule    — warmup + cosine decay (Task 11)
  Trainer               — training loop with AdamW (Task 12)

[theory] GPT-2 is trained with a standard language modelling objective:
maximise the log-likelihood of each token given all previous tokens in the
sequence. Equivalently, minimise the cross-entropy loss averaged over all
token positions.

[best practice] Separating the loss function, LR schedule, and training loop
into distinct units makes each component individually testable and reusable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from gpt2.model import GPT2Model

# ===========================================================================
# Task 10 — Language Modelling Loss
# ===========================================================================

# [theory] Next-token prediction: given input tokens [t₀, t₁, …, t_{T-1}],
# the model should predict [t₁, t₂, …, t_T].
#
# Concretely:
#   inputs  = token_ids[:, :-1]     shape: (B, T-1)
#   targets = token_ids[:, 1:]      shape: (B, T-1)
#
# For each position i, the model outputs logits over the vocabulary, and we
# compute cross-entropy against the ground-truth next token.
#
# Cross-entropy for a single position:
#   loss_i = −log( softmax(logits_i)[target_i] )
#          = −logits_i[target_i] + log(∑_v exp(logits_i[v]))
#
# The total loss is the mean across all positions and batch elements.

# Warmup -----------------------------------------------------------------------
# Consider the sentence "Hello World" tokenized as [15496, 2159].
# When we train on this sequence:
#   inputs  = [15496]   ("Hello")
#   targets = [2159]    ("World")
# The model sees "Hello" and must predict "World".
#
# Q: For a sequence [A, B, C, D], write out the (input, target) pairs:
#   input  = [?, ?, ?]
#   target = [?, ?, ?]
#
# Q: Why do we NOT include the last token in inputs? And why not the first
#    token in targets?
# ------------------------------------------------------------------------------


def cross_entropy_loss(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute next-token prediction cross-entropy loss.

    Parameters
    ----------
    logits : torch.Tensor
        Model output, shape (B, T, vocab_size). These are the raw unnormalised
        scores BEFORE softmax.
    token_ids : torch.Tensor
        Ground-truth token IDs, shape (B, T). Integer dtype.

    Returns
    -------
    torch.Tensor
        Scalar loss value (mean over all positions and batch elements).

    Steps
    -----
    1. Slice: inputs logits at positions 0..T-2  → (B, T-1, vocab_size)
              targets token_ids at positions 1..T-1 → (B, T-1)
    2. Flatten: logits → (B*(T-1), vocab_size), targets → (B*(T-1),)
    3. Return F.cross_entropy(logits_flat, targets_flat)
    """
    raise NotImplementedError(
        "Task 10: Implement cross_entropy_loss.\n"
        "  logits_shifted  = logits[:, :-1, :].contiguous()\n"
        "  targets_shifted = token_ids[:, 1:].contiguous()\n"
        "  return F.cross_entropy(logits_shifted.view(-1, logits.size(-1)),\n"
        "                         targets_shifted.view(-1))"
    )


# ===========================================================================
# Task 11 — Learning Rate Schedule
# ===========================================================================

# [theory] GPT-2 uses a cosine annealing schedule with linear warmup (§3):
#
#   Phase 1 — Linear warmup (steps 0 → warmup_steps):
#     lr(t) = max_lr · t / warmup_steps
#
#   Phase 2 — Cosine decay (steps warmup_steps → max_steps):
#     progress = (t − warmup_steps) / (max_steps − warmup_steps)
#     lr(t) = min_lr + 0.5 · (max_lr − min_lr) · (1 + cos(π · progress))
#
#   After max_steps, lr is held at min_lr.
#
# The cosine schedule is smooth and avoids the hard LR drop of step decay.
# Loshchilov & Hutter (2017) popularised this for deep learning (SGDR).
#
# [modern] Recent work (Hoffmann et al., Chinchilla) uses a cosine schedule as
# well. An emerging alternative is Warmup-Stable-Decay (WSD): train at constant
# LR then decay steeply at the end, which allows extending training cheaply.

# Warmup -----------------------------------------------------------------------
# Sketch the LR curve for: max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000.
#
#   step=0:     lr = ?
#   step=50:    lr = ?   (halfway through warmup)
#   step=100:   lr = ?   (end of warmup, start of cosine)
#   step=550:   lr = ?   (halfway through cosine decay)
#   step=1000:  lr = ?   (end of training)
#
# Expected: 0.0 → 5e-4 → 1e-3 → ~5.5e-4 → 1e-4
# ------------------------------------------------------------------------------


def cosine_lr_schedule(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """
    Compute the learning rate at a given training step.

    Parameters
    ----------
    step : int
        Current training step (0-indexed).
    max_lr : float
        Peak learning rate reached at the end of warmup.
    min_lr : float
        Minimum learning rate held after cosine decay completes.
        The GPT-2 paper uses min_lr = max_lr / 10.
    warmup_steps : int
        Number of linear warmup steps.
    max_steps : int
        Total number of training steps (including warmup).

    Returns
    -------
    float
        Learning rate for this step.
    """
    raise NotImplementedError(
        "Task 11: Implement cosine_lr_schedule.\n"
        "  Phase 1 (step < warmup_steps):  return max_lr * step / warmup_steps\n"
        "  Phase 2 (step <= max_steps):    cosine decay toward min_lr\n"
        "    progress = (step - warmup_steps) / (max_steps - warmup_steps)\n"
        "    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))\n"
        "    return min_lr + coeff * (max_lr - min_lr)\n"
        "  Phase 3 (step > max_steps):     return min_lr"
    )


# ===========================================================================
# Task 12 — Training Loop (Trainer)
# ===========================================================================

# [theory] GPT-2 is trained with AdamW (Loshchilov & Hutter, 2019), which
# decouples L2 regularisation from the adaptive moment estimates:
#
#   θ_t = θ_{t-1} − lr · (Adam update + λ · θ_{t-1})
#
# Key hyperparameters from the paper:
#   β₁ = 0.9, β₂ = 0.999, ε = 1e-8, weight_decay = 0.01
#   gradient clipping: max norm = 1.0
#
# Gradient clipping prevents "exploding gradients" by rescaling the gradient
# vector whenever its global L2 norm exceeds the threshold. This keeps training
# stable, especially early on when the model is far from its optimum.
#
# [best practice] Always zero gradients BEFORE the forward pass (not after
# the backward pass) to avoid subtle bugs when code paths are conditional.
# Use optimizer.zero_grad(set_to_none=True) — it's faster than zero_grad()
# because it frees the gradient tensors rather than filling them with zeros.
#
# [modern] Large-scale training adds: gradient accumulation (to simulate larger
# batches), mixed-precision (bf16 forward + fp32 master weights), and
# distributed training (DDP or FSDP). None of these are implemented here.

# Warmup -----------------------------------------------------------------------
# Order the following training operations correctly (1 = first, 5 = last):
#   ___ optimizer.step()
#   ___ loss = cross_entropy_loss(logits, batch)
#   ___ optimizer.zero_grad(set_to_none=True)
#   ___ logits = model(batch)
#   ___ loss.backward()
#   ___ torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#
# Answer: 3, 2, 1, 4, 5, 6  (zero → forward → loss → backward → clip → step)
# ------------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """
    Hyperparameters for the training loop.

    [theory] These values are from the GPT-2 paper §3 (small model):
      • batch_size=512 sequences × 1024 tokens = ~524K tokens per step
      • We use much smaller values by default for local experimentation.
    """

    max_steps: int = 10_000
    batch_size: int = 8
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1           # min_lr = learning_rate * min_lr_ratio
    warmup_steps: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Optional: log loss every this many steps (0 = never)
    log_interval: int = 100

    # Device is resolved at runtime; override for tests
    device: str = field(default="cpu")


class Trainer:
    """
    Stateful training loop for GPT-2.

    Usage
    -----
    >>> trainer = Trainer(model, config)
    >>> for step, loss in trainer.train(data_iter):
    ...     if step % 100 == 0:
    ...         print(f"step {step}: loss {loss:.4f}")
    """

    def __init__(self, model: GPT2Model, config: TrainConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = self._build_optimizer()
        self.step = 0

    def _build_optimizer(self) -> torch.optim.AdamW:
        """
        Create AdamW with weight decay applied only to 2-D weight tensors.

        [best practice] It is standard practice to NOT apply weight decay to
        1-D parameters (biases, LayerNorm weight/bias). These have far fewer
        parameters and regularising them can harm training.
        """
        decay_params = [p for p in self.model.parameters() if p.dim() >= 2]
        nodecay_params = [p for p in self.model.parameters() if p.dim() < 2]
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

    def _update_lr(self) -> float:
        """Compute and apply the current LR to all parameter groups."""
        lr = cosine_lr_schedule(
            step=self.step,
            max_lr=self.config.learning_rate,
            min_lr=self.config.learning_rate * self.config.min_lr_ratio,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def train_step(self, token_ids: torch.Tensor) -> float:
        """
        Run one gradient update step.

        Parameters
        ----------
        token_ids : torch.Tensor
            Integer tensor of shape (B, T+1). The extra token allows us to
            form T (input, target) pairs by shifting by one position.

        Returns
        -------
        float
            The scalar loss value for this step.

        Steps
        -----
        1. Zero gradients (set_to_none=True).
        2. Forward pass: logits = model(token_ids).
        3. Compute loss = cross_entropy_loss(logits, token_ids).
        4. Backward pass: loss.backward().
        5. Clip gradients: torch.nn.utils.clip_grad_norm_(params, grad_clip).
        6. Update LR via _update_lr().
        7. Optimizer step.
        8. Increment self.step.
        9. Return loss.item().
        """
        raise NotImplementedError(
            "Task 12: Implement Trainer.train_step.\n"
            "  Follow the 9-step order above exactly.\n"
            "  Use: self.optimizer.zero_grad(set_to_none=True)\n"
            "       torch.nn.utils.clip_grad_norm_(\n"
            "           self.model.parameters(), self.config.grad_clip)\n"
            "       self._update_lr()"
        )

    def train(self, data_iter: Iterator[torch.Tensor]) -> Iterator[tuple[int, float]]:
        """
        Run the full training loop, yielding (step, loss) pairs.

        Parameters
        ----------
        data_iter : Iterator[torch.Tensor]
            An iterator that yields batches of token IDs, shape (B, T+1).
            It should produce batches indefinitely (or raise StopIteration to
            end training early).

        Yields
        ------
        tuple[int, float]
            (step_number, loss_value) for every step up to max_steps.
        """
        self.model.train()
        for token_ids in data_iter:
            if self.step >= self.config.max_steps:
                break
            token_ids = token_ids.to(self.config.device)
            loss = self.train_step(token_ids)
            yield self.step, loss
