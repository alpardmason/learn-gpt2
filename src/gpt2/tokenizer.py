"""
Tokenizers for GPT-2 reimplementation.

Two tokenizers are provided:

  CharTokenizer  — character-level tokenizer; fully implemented; used in all tests
                   because it requires no external vocab files.
  BPETokenizer   — byte-pair encoding tokenizer; stub wrapping tiktoken; YOUR TASK.

[theory] Tokenization converts raw text into integer IDs that the model processes.
GPT-2 uses Byte-Pair Encoding (BPE), which starts from individual bytes and
iteratively merges the most frequent adjacent pairs into new tokens. This gives a
vocabulary that handles any Unicode text without UNK tokens.

The key insight: the vocabulary is learned from data, not hand-crafted. A word like
"running" might become ["run", "ning"] or ["runn", "ing"] depending on what merges
were most frequent in the training corpus.

[modern] Modern models (GPT-4, LLaMA 3, Gemma) use SentencePiece or tiktoken with
similar BPE logic but different vocabulary sizes (32K–128K). GPT-2's 50,257-token
vocabulary is on the smaller end.

Reference: references/gpt-2/src/encoder.py (OpenAI's original BPE implementation)
"""

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract base class for all tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integer token IDs."""
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer token IDs back to a string."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of distinct token types."""
        ...


# ---------------------------------------------------------------------------
# CharTokenizer — fully implemented, used in all tests
# ---------------------------------------------------------------------------


class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer.

    Builds a vocabulary from all unique characters in a corpus (or from an
    explicit character list). Each character maps to a unique integer ID.

    This tokenizer is intentionally simple so that test fixtures do not depend
    on external vocabulary files. It is NOT used for real GPT-2 training.

    [best practice] Test fixtures should be self-contained and fast. A char
    tokenizer eliminates the need to load a 1 MB vocab file for every test.
    """

    def __init__(self, chars: str | list[str] | None = None) -> None:
        """
        Parameters
        ----------
        chars : str | list[str] | None
            The set of characters that define the vocabulary. If None, a default
            printable ASCII vocabulary is used (96 characters + padding to 128).
        """
        if chars is None:
            # Default: printable ASCII (space through ~) for general use in tests
            chars = [chr(i) for i in range(32, 128)]
        elif isinstance(chars, str):
            # Build sorted unique vocab from the string itself
            chars = sorted(set(chars))
        else:
            chars = sorted(set(chars))

        self._stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self._itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """
        Build a CharTokenizer whose vocabulary covers every character in `text`.

        Example
        -------
        >>> tok = CharTokenizer.from_text("hello world")
        >>> tok.vocab_size
        8
        >>> tok.encode("hello")
        [1, 2, 3, 3, 4]
        """
        return cls(chars=sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of integer token IDs.

        Raises
        ------
        KeyError
            If the text contains a character not in the vocabulary.
        """
        return [self._stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Decode a list of integer token IDs back to a string."""
        return "".join(self._itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)


# ---------------------------------------------------------------------------
# BPETokenizer — YOUR TASK (Task 3)
# ---------------------------------------------------------------------------

# [theory] Byte-Pair Encoding (Sennrich et al., 2016):
#   1. Start with a vocabulary of individual bytes (256 tokens).
#   2. Count all adjacent token pairs in the corpus.
#   3. Merge the most frequent pair into a new token.
#   4. Repeat until the vocabulary reaches the target size (50,000 merges for GPT-2).
#
# The result is a vocabulary that efficiently represents common words as single tokens
# while rare words are split into subword pieces — a good compression/coverage tradeoff.
#
# GPT-2 uses a byte-level BPE: the 256 raw bytes are the initial vocabulary, so the
# tokenizer never produces UNK tokens even for emoji or non-Latin scripts.
#
# Reference: references/gpt-2/src/encoder.py
#            https://huggingface.co/docs/tokenizers/quicktour

# Warmup -----------------------------------------------------------------------
# Before implementing, answer these questions by hand:
#
# Q1. Suppose you have the corpus: "aab aab ab b" and you run one BPE merge step.
#     Which pair gets merged? What does the vocabulary look like after the merge?
#
# Q2. The GPT-2 vocab has 50,257 tokens: 50,000 BPE merges + 256 byte tokens + 1
#     special token. Which special token is that, and what is it used for?
#
# Q3. Why does tiktoken's encode() return different IDs for "Hello" vs " Hello"
#     (with a leading space)? What does this tell you about GPT-2's tokenization?
# ------------------------------------------------------------------------------


class BPETokenizer(Tokenizer):
    """
    Byte-Pair Encoding tokenizer wrapping tiktoken's GPT-2 encoding.

    tiktoken is OpenAI's fast BPE implementation in Rust, used in production.
    For educational purposes, you will implement a thin Python wrapper around
    tiktoken so you understand the encode/decode interface.

    [modern] Modern tokenizers like tiktoken and sentencepiece are implemented in
    compiled languages (Rust/C++) for speed. A pure-Python BPE tokenizer would be
    100–1000× slower, which matters when preprocessing billions of tokens.

    Task
    ----
    Implement __init__, encode, and decode using tiktoken:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        enc.encode(text)   -> list[int]
        enc.decode(ids)    -> str  (note: tiktoken decode takes list[int])
    """

    def __init__(self) -> None:
        # Warmup: what is tiktoken.get_encoding("gpt2").n_vocab? Verify it equals
        # the vocab_size you expect from the GPT-2 paper.
        raise NotImplementedError(
            "Task 3: Implement BPETokenizer.__init__.\n"
            "  Hint: import tiktoken; self._enc = tiktoken.get_encoding('gpt2')"
        )

    def encode(self, text: str) -> list[int]:
        # Warmup: call enc.encode("Hello, world!") in a Python REPL.
        # How many tokens does "Hello, world!" produce? What are their IDs?
        raise NotImplementedError(
            "Task 3: Implement BPETokenizer.encode.\n"
            "  Hint: return self._enc.encode(text)"
        )

    def decode(self, ids: list[int]) -> str:
        # Note: tiktoken's decode method accepts list[int] directly.
        raise NotImplementedError(
            "Task 3: Implement BPETokenizer.decode.\n"
            "  Hint: return self._enc.decode(ids)"
        )

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError(
            "Task 3: Implement BPETokenizer.vocab_size.\n"
            "  Hint: return self._enc.n_vocab"
        )
