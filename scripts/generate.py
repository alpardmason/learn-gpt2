"""
CLI entry point for GPT-2 text generation.

Usage
-----
uv run python scripts/generate.py --prompt "Once upon a time"
uv run python scripts/generate.py --prompt "Hello" --max_new_tokens 50 --top_k 40
uv run python scripts/generate.py --checkpoint checkpoints/gpt2_small_step10000.pt

NOTE: Without a trained checkpoint the model will generate random tokens
(the model is randomly initialised). Always train first with scripts/train.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from gpt2.config import GPT2_TINY, GPT2Config
from gpt2.generate import generate
from gpt2.model import GPT2Model
from gpt2.tokenizer import BPETokenizer, CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT-2 model")
    parser.add_argument("--prompt", type=str, default="Hello",
                        help="Prompt string to condition generation on")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to a .pt checkpoint saved by scripts/train.py")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint is not None:
        # Load from checkpoint
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        config: GPT2Config = ckpt["config"]
        model = GPT2Model(config)
        model.load_state_dict(ckpt["model"])
        tokenizer = BPETokenizer()
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        # No checkpoint: use tiny model + char tokenizer for demo
        print("No checkpoint provided — using tiny random model with CharTokenizer.")
        config = GPT2_TINY
        model = GPT2Model(config)
        tokenizer = CharTokenizer()

    model.eval().to(args.device)

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Generating {args.max_new_tokens} new tokens...\n")

    out_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )

    # Decode only the newly generated tokens
    new_ids = out_ids[0, len(prompt_ids):].tolist()
    generated_text = tokenizer.decode(new_ids)

    print(args.prompt + generated_text)


if __name__ == "__main__":
    main()
