"""
Tokenizers for GPT-2 reimplementation.

Two tokenizers are provided:

  CharTokenizer  — character-level tokenizer; fully implemented; used in all tests
                   because it requires no external vocab files.
  BPETokenizer   — byte-pair encoding tokenizer; from-scratch stub; YOUR TASK.

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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter


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
    def from_text(cls, text: str) -> CharTokenizer:
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
# Q3. After training on "aaab aaab ab b" with 3 merges, how many token IDs
#     does encode("aaab") produce? Why is it fewer than 4?
# ------------------------------------------------------------------------------


class BPETokenizer(Tokenizer):
    """
    Byte-level BPE tokenizer implemented from scratch.

    You will implement the full training and encode/decode pipeline — no
    tiktoken wrapper. Start from 256 byte tokens, learn merges from a corpus,
    then greedily apply those merges at encode time.

    [modern] Production tokenizers (tiktoken, SentencePiece) use the same BPE
    logic but are compiled (Rust/C++) for speed. A pure-Python implementation
    is fine for learning; it would be 100–1000× slower on billion-token corpora.

    Reference: OpenAI's original implementation in references/gpt-2/src/encoder.py
    """

    ENDOFTEXT = "<|endoftext|>"

    def __init__(self) -> None:
        """
        Initialize an untrained tokenizer with the 256 raw byte tokens.

        Set up:
          self._id_to_bytes: dict[int, bytes]  — token ID → byte sequence
          self._merges: list[tuple[int, int, int]] — ordered merge rules (left, right, new_id)
        """
        self._id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._merges: list[tuple[int, int, int]] = []

    @classmethod
    def from_corpus(cls, corpus: str, num_merges: int) -> BPETokenizer:
        """
        Train a new BPETokenizer on `corpus` and return it.

        Convenience wrapper: construct, call train(), return.
        """
        tokenizer = BPETokenizer()
        tokenizer.train(corpus, num_merges)
        return tokenizer

    def train(self, corpus: str, num_merges: int) -> None:
        """
        Learn BPE merge rules from `corpus`.

        [theory] Repeat num_merges times:
          1. Convert corpus to byte IDs via _text_to_byte_ids.
          2. Count adjacent pairs with _get_pair_counts.
          3. Pick the most frequent pair; assign it the next token ID.
          4. Record the merge and update self._id_to_bytes for the new token.
        """
        byte_ids = self._text_to_byte_ids(corpus)

        for _ in range(num_merges):
            # If the sequence is too short
            if len(byte_ids) <= 1:
                break

            # Frequency statistics:
            counts = self._get_pair_counts(byte_ids)
            if not counts:
                break

            # Pick the most frequent pair.
            # Original: pair = max(counts, key=lambda p: counts[p])
            # [best practice] Ties must break deterministically. Count-only max picks whichever
            # pair happened to be inserted first in the dict, which can vary across runs.
            # max on (count, pair) picks highest frequency; on ties, lexicographically largest pair.
            pair = max(counts, key=lambda p: (counts[p], p))

            # Assign ID:
            new_id = len(self._id_to_bytes)

            self._id_to_bytes[new_id] = self._id_to_bytes[pair[0]] + self._id_to_bytes[pair[1]]

            # Original: self._merges.append((pair[0], pair[1]))
            # [best practice] Store new_id with the rule so encode() replays the exact ID from
            # training instead of recomputing 256 + rank (fragile if training stops early).
            self._merges.append((pair[0], pair[1], new_id))

            # Merge new id:
            byte_ids = self._merge_pair(byte_ids, pair, new_id)

    def _text_to_byte_ids(self, text: str) -> list[int]:
        """Convert UTF-8 text to initial byte token IDs (0–255)."""
        # ![CAUTION] UTF-8 encoding and decoding!!!
        return list(text.encode("utf-8"))

    @staticmethod
    def _get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
        """Count adjacent token-ID pairs in a sequence."""
        if len(ids) < 2:
            return {}

        # Original: manual dict loop with counts.get((i, j), 0) + 1
        # [best practice] Counter tallies in C; same O(n) complexity but faster and clearer.
        return dict(Counter(zip(ids, ids[1:])))

    @staticmethod
    def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """
        Replace all non-overlapping occurrences of `pair` with `new_id`.

        Scan left-to-right; when (ids[i], ids[i+1]) == pair, emit new_id
        and advance i by 2. Otherwise emit ids[i] and advance by 1.
        """
        merged: list[int] = []
        i = 0
        n = len(ids)
        # Original: compared ids[i] and ids[i + 1] separately and used i = i + 2 / i = i + 1
        # [best practice] Cache n once; compare (ids[i], ids[i + 1]) as a tuple — same logic,
        # fewer repeated bounds checks and more idiomatic pair comparison.
        while i < n:
            if i + 1 < n and (ids[i], ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def _apply_merges(self, ids: list[int]) -> list[int]:
        """
        Greedily apply learned merges to a byte-ID sequence.

        Apply merges in training order (rank 0 first). After each merge rule,
        re-scan for the next rule until all merges are attempted.
        """
        # Original: for rank, (a, b) in enumerate(self._merges):
        #               ids = self._merge_pair(ids, (a, b), 256 + rank)
        # [best practice] Use the new_id recorded during training; 256 + rank assumes every
        # merge rank was filled sequentially with no gaps.
        for a, b, new_id in self._merges:
            ids = self._merge_pair(ids, (a, b), new_id)

        return ids

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integer token IDs."""
        byte_ids = self._text_to_byte_ids(text)

        return self._apply_merges(byte_ids)

    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer token IDs back to a string."""
        # ![CAUTION] raw bytes manipulation!
        return b"".join(self._id_to_bytes[i] for i in ids).decode("utf-8")

    @property
    def vocab_size(self) -> int:
        """Number of distinct token types (256 bytes + learned merges)."""
        return len(self._id_to_bytes)
