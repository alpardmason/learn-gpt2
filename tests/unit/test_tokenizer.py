"""
Unit tests for tokenizers.

CharTokenizer is fully tested (it's a complete implementation).
BPETokenizer tests are skipped if the implementation is still a stub.
"""

import pytest

from gpt2.tokenizer import BPETokenizer, CharTokenizer


class TestCharTokenizerRoundTrip:
    def test_encode_decode_roundtrip(self) -> None:
        tok = CharTokenizer.from_text("hello world")
        text = "hello"
        assert tok.decode(tok.encode(text)) == text

    def test_full_string_roundtrip(self) -> None:
        corpus = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"
        tok = CharTokenizer.from_text(corpus)
        assert tok.decode(tok.encode(corpus)) == corpus

    def test_single_char(self) -> None:
        tok = CharTokenizer.from_text("a")
        assert tok.encode("a") == [0]
        assert tok.decode([0]) == "a"


class TestCharTokenizerVocab:
    def test_vocab_size_from_text(self) -> None:
        tok = CharTokenizer.from_text("aab")
        assert tok.vocab_size == 2  # {'a', 'b'}

    def test_default_vocab_covers_ascii(self) -> None:
        tok = CharTokenizer()
        # All printable ASCII (32–127) should be covered
        for c in "Hello, World! 123":
            assert c in tok._stoi

    def test_ids_are_unique(self) -> None:
        tok = CharTokenizer.from_text("abcdef")
        ids = [tok._stoi[c] for c in "abcdef"]
        assert len(ids) == len(set(ids))

    def test_stoi_itos_invertible(self) -> None:
        tok = CharTokenizer.from_text("hello")
        for ch, idx in tok._stoi.items():
            assert tok._itos[idx] == ch


class TestCharTokenizerErrors:
    def test_encode_unknown_char_raises(self) -> None:
        tok = CharTokenizer.from_text("abc")
        with pytest.raises(KeyError):
            tok.encode("z")  # 'z' is not in the vocab


class TestBPETokenizer:
    """
    BPETokenizer tests are skipped if the stub is not yet implemented.
    These tests will be unblocked automatically once you complete Task 3.
    """

    _CORPUS = "aaab aaab ab b " * 20
    _NUM_MERGES = 10

    @pytest.fixture()
    def bpe(self) -> BPETokenizer:
        try:
            return BPETokenizer.from_corpus(self._CORPUS, num_merges=self._NUM_MERGES)
        except NotImplementedError:
            pytest.skip("BPETokenizer not yet implemented (Task 3)")

    def test_vocab_size(self, bpe: BPETokenizer) -> None:
        assert bpe.vocab_size == 256 + self._NUM_MERGES

    def test_encode_returns_ints(self, bpe: BPETokenizer) -> None:
        ids = bpe.encode("aaab")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode_roundtrip(self, bpe: BPETokenizer) -> None:
        text = "aaab ab"
        assert bpe.decode(bpe.encode(text)) == text

    def test_merges_reduce_token_count(self, bpe: BPETokenizer) -> None:
        """After training, frequent byte sequences should compress to fewer tokens."""
        raw_byte_count = len(b"aaab")
        merged_count = len(bpe.encode("aaab"))
        assert merged_count < raw_byte_count
