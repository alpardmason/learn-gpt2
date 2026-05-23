"""
GPT-2 model configuration.

[theory] The GPT-2 paper (Table 2) defines four model sizes that all share the same
architecture but differ in depth (n_layer), width (n_embd), and number of attention
heads (n_head). The context window is always 1024 tokens and the vocabulary is always
50,257 tokens (BPE). We express each size as a named preset below.

[best practice] Using a frozen dataclass as a configuration object is a modern Python
pattern: it is type-annotated, immutable, serialisable to JSON/YAML, and self-documenting.
Compare to the original OpenAI code which uses HParams from TensorFlow.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPT2Config:
    """
    Hyperparameters for a single GPT-2 variant.

    Attributes
    ----------
    vocab_size : int
        Number of BPE token types. GPT-2 uses 50,257 (50,000 merges + 256 byte
        tokens + 1 special <|endoftext|> token).
    n_ctx : int
        Maximum sequence length (context window). GPT-2 uses 1024.
    n_embd : int
        Embedding / hidden dimension d_model. All projections use this width.
    n_head : int
        Number of attention heads. Must evenly divide n_embd so that each head
        has dimension head_dim = n_embd // n_head.
    n_layer : int
        Number of stacked transformer blocks.

    [theory] The model parameter count (excluding embeddings) scales roughly as
    12 * n_layer * n_embd^2. The four published sizes are:
        small  (117M):  n_layer=12,  n_embd=768,  n_head=12
        medium (345M):  n_layer=24,  n_embd=1024, n_head=16
        large  (762M):  n_layer=36,  n_embd=1280, n_head=20
        xl    (1542M):  n_layer=48,  n_embd=1600, n_head=25
    """

    vocab_size: int = 50257
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head's Q/K/V vectors."""
        return self.n_embd // self.n_head


# ---------------------------------------------------------------------------
# Named presets — all four model sizes from GPT-2 paper Table 2
# ---------------------------------------------------------------------------

# [theory] "Small" is the model used in most educational contexts. It has 117M
# parameters total (124M including embeddings), fits on a single consumer GPU,
# and is the variant whose weights were originally released.
GPT2_SMALL = GPT2Config(
    vocab_size=50257,
    n_ctx=1024,
    n_embd=768,
    n_head=12,
    n_layer=12,
)

GPT2_MEDIUM = GPT2Config(
    vocab_size=50257,
    n_ctx=1024,
    n_embd=1024,
    n_head=16,
    n_layer=24,
)

GPT2_LARGE = GPT2Config(
    vocab_size=50257,
    n_ctx=1024,
    n_embd=1280,
    n_head=20,
    n_layer=36,
)

GPT2_XL = GPT2Config(
    vocab_size=50257,
    n_ctx=1024,
    n_embd=1600,
    n_head=25,
    n_layer=48,
)

# Tiny config used throughout the test suite — fast on CPU, deterministic.
# [best practice] Always define a "tiny" config for unit tests so they run in
# milliseconds without a GPU. This is standard practice in ML research codebases.
GPT2_TINY = GPT2Config(
    vocab_size=64,
    n_ctx=16,
    n_embd=32,
    n_head=4,
    n_layer=2,
)
