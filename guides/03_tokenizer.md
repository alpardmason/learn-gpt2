# Guide 03 — Tokenizer (BPE from Scratch)

## Theory

**Byte-Pair Encoding (BPE)** (Sennrich et al., 2016) builds a vocabulary by
iteratively merging the most frequent pair of adjacent tokens:

```
Initial:   "l o w e r"  "n e w e r"  "w i d e r"    (space separated chars)
Step 1:    merge ("e","r") → "er"
           "l o w er"  "n e w er"  "w i d er"
Step 2:    merge ("w","er") → "wer"
           "l o wer"  "n e wer"  "w i d er"
... and so on until the vocabulary reaches the target size.
```

**GPT-2 uses byte-level BPE:**
- Start from 256 raw byte tokens (handles any Unicode, no `<UNK>` needed)
- Perform 50,000 merges → vocabulary of 50,256 tokens
- Add 1 special token `<|endoftext|>` → **50,257 total**

The `<|endoftext|>` token is inserted between documents in the training corpus
so the model learns where documents begin and end.

Production GPT-2 also applies a regex pre-tokenizer before BPE (see OpenAI's
`encoder.py`) so merges respect word boundaries. Your task starts with a simpler
byte-level pipeline; the full GPT-2 regex is an optional stretch goal.

**Why not character-level or word-level?**
- Word-level: vocabulary explodes for morphologically rich languages; can't
  handle unknown words (OOV problem).
- Character-level: very long sequences; the model must learn to compose
  characters into words in attention — expensive.
- BPE hits the sweet spot: common words are single tokens, rare words are
  split into meaningful subwords.

## Warmup

1. Given corpus `"aab aab ab b"` and vocabulary starting with `{a, b, ' '}`:
   - Count all adjacent pairs. Which pair is most frequent?
   - After one merge, what does the vocabulary look like?

2. Why does byte-level BPE never need an `<UNK>` token?

3. After training on `"aaab aaab ab b" * 20` with 3 merges, encode `"aaab"`.
   How many token IDs do you get? Fewer than 4? Why?

## Your Task

Open `src/gpt2/tokenizer.py` and implement `BPETokenizer` **from scratch** —
no `tiktoken` wrapper. Follow the algorithm in the stub comments.

### Algorithm overview

**Training** (`train` / `from_corpus`):

1. Convert the corpus to a list of byte token IDs (0–255).
2. Count adjacent `(id, id)` pairs across the corpus.
3. Merge the most frequent pair into a new token ID (256, 257, …).
4. Repeat for `num_merges` steps.
5. Store merge rules in order — merge rank matters at encode time.

**Encoding** (`encode`):

1. Convert text to byte IDs (0–255).
2. Greedily apply learned merges in the order they were learned.
3. Return the final token ID list.

**Decoding** (`decode`):

1. Map each token ID back to its byte sequence.
2. Concatenate bytes and decode as UTF-8.

### Methods to implement

```
class BPETokenizer(Tokenizer):
    def __init__(self) -> None:                          ← Task 3a
    def from_corpus(cls, corpus, num_merges) -> Self:    ← Task 3b
    def train(self, corpus, num_merges) -> None:          ← Task 3c
    def _text_to_byte_ids(self, text) -> list[int]:      ← Task 3d
    def _get_pair_counts(self, ids) -> dict:             ← Task 3e
    def _merge_pair(self, ids, pair, new_id) -> list:    ← Task 3f
    def _apply_merges(self, ids) -> list[int]:           ← Task 3g
    def encode(self, text) -> list[int]:                 ← Task 3h
    def decode(self, ids) -> str:                         ← Task 3i
    def vocab_size(self) -> int:                          ← Task 3j (property)
```

**Hints:**
- Start with `_text_to_byte_ids`: `list(text.encode("utf-8"))` gives byte values 0–255.
- `_merge_pair` must scan left-to-right and skip overlapping matches (see stub).
- `_apply_merges` applies merges in **training order** (lowest rank first), same as OpenAI's `Encoder.bpe`.
- Use `BPETokenizer.from_corpus(corpus, num_merges)` in tests — do not hard-code GPT-2's 50k merges.

## Example Input / Output

```python
corpus = "aaab aaab ab b " * 20
tok = BPETokenizer.from_corpus(corpus, num_merges=3)

print(tok.vocab_size)               # 259  (256 bytes + 3 merges)
ids = tok.encode("aaab")
print(len(ids))                     # fewer than 4 after merges fire
print(tok.decode(ids))              # "aaab"
assert tok.decode(tok.encode("aaab")) == "aaab"
```

To compare your implementation against production GPT-2 tokenization (optional):

```python
import tiktoken
ref = tiktoken.get_encoding("gpt2")
ref.encode("Hello, world!")   # reference IDs — yours will differ unless you load GPT-2 merges
```

## Modern LLM Comparison

| Model | Tokenizer | Vocab Size | Notes |
|-------|-----------|------------|-------|
| GPT-2 | BPE (byte-level) | 50,257 | regex pre-tokenizer + 50k merges |
| GPT-4 | cl100k\_base | 100,277 | larger vocab → fewer tokens/doc |
| LLaMA 3 | tiktoken | 128,256 | even larger vocab |
| LLaMA 2 | SentencePiece | 32,000 | unigram LM, not BPE |

Production tokenizers (tiktoken, SentencePiece) are implemented in Rust/C++ for
speed — a pure-Python BPE is fine for learning but would be 100–1000× slower
on billion-token corpora.

Larger vocabularies encode text more efficiently (fewer tokens per sentence),
which reduces sequence length and allows more text to fit in the context window.

## Run the Tests

```bash
uv run pytest tests/unit/test_tokenizer.py -v
```

`BPETokenizer` tests are automatically skipped until you complete this task.
`CharTokenizer` tests run immediately and should all pass.
