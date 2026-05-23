"""
CLI entry point for GPT-2 training.

Usage
-----
uv run python scripts/train.py
uv run python scripts/train.py --max_steps 5000 --batch_size 4
uv run python scripts/train.py --preset medium --device cuda

The script trains on a simple synthetic repeated-sequence dataset by default,
which is enough to verify that your implementation learns (loss decreases).
For real training, replace `synthetic_data_iter` with your own data loader.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from gpt2.config import GPT2_LARGE, GPT2_MEDIUM, GPT2_SMALL, GPT2_TINY, GPT2_XL, GPT2Config
from gpt2.model import GPT2Model
from gpt2.train import TrainConfig, Trainer

PRESETS: dict[str, GPT2Config] = {
    "tiny": GPT2_TINY,
    "small": GPT2_SMALL,
    "medium": GPT2_MEDIUM,
    "large": GPT2_LARGE,
    "xl": GPT2_XL,
}


def synthetic_data_iter(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    device: str,
    seed: int = 42,
) -> torch.Tensor:
    """
    Infinite iterator of random token batches.

    Yields batches of shape (batch_size, seq_len + 1). The +1 allows
    cross_entropy_loss to form (input, target) pairs by shifting by one.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    while True:
        yield torch.randint(
            0, vocab_size, (batch_size, seq_len + 1),
            generator=rng, device=device,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GPT-2 model")
    parser.add_argument("--preset", choices=list(PRESETS), default="tiny",
                        help="Model size preset (default: tiny)")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    model_config = PRESETS[args.preset]
    train_config = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
        device=args.device,
    )

    print(f"Model preset : {args.preset}")
    print(f"Parameters   : {GPT2Model(model_config).num_parameters():,}")
    print(f"Device       : {args.device}")
    print(f"Max steps    : {args.max_steps}")
    print()

    model = GPT2Model(model_config)
    trainer = Trainer(model, train_config)

    data = synthetic_data_iter(
        vocab_size=model_config.vocab_size,
        seq_len=model_config.n_ctx,
        batch_size=args.batch_size,
        device=args.device,
    )

    for step, loss in trainer.train(data):
        if step % args.log_interval == 0:
            print(f"step {step:>6d} | loss {loss:.4f}")

    # Save final checkpoint
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.checkpoint_dir / f"gpt2_{args.preset}_step{args.max_steps}.pt"
    torch.save({"model": model.state_dict(), "config": model_config}, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
