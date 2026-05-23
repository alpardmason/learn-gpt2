# Guide 03 — Tokenizer (BPE)

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

2. Run `tiktoken.get_encoding("gpt2").encode("Hello")` and
   `tiktoken.get_encoding("gpt2").encode(" Hello")` in a Python REPL.
   Do they produce different IDs? Why?

3. What is `tiktoken.get_encoding("gpt2").n_vocab`? Verify it equals 50257.

## Your Task

Open `src/gpt2/tokenizer.py` and implement `BPETokenizer`:

```
class BPETokenizer(Tokenizer):
    def __init__(self) -> None:   ← Task 3
    def encode(self, text) -> list[int]:   ← Task 3
    def decode(self, ids) -> str:   ← Task 3
    def vocab_size(self) -> int:   ← Task 3 (property)
```

**Hint:** The implementation is intentionally short — three lines of tiktoken calls.

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
enc.encode("Hello, world!")   # → list[int]
enc.decode([15496, 11, 995])  # → str
enc.n_vocab                   # → 50257
```

## Example Input / Output

```python
tok = BPETokenizer()
print(tok.vocab_size)               # 50257
ids = tok.encode("Hello, world!")   # [15496, 11, 995, 0]
print(ids)                          # [15496, 11, 995, 0]
print(tok.decode(ids))              # "Hello, world!"
# Round-trip works
assert tok.decode(tok.encode("The quick brown fox")) == "The quick brown fox"
```

## Modern LLM Comparison

| Model | Tokenizer | Vocab Size | Notes |
|-------|-----------|------------|-------|
| GPT-2 | BPE (tiktoken) | 50,257 | byte-level |
| GPT-4 | cl100k\_base | 100,277 | larger vocab → fewer tokens/doc |
| LLaMA 3 | tiktoken | 128,256 | even larger vocab |
| LLaMA 2 | SentencePiece | 32,000 | unigram LM, not BPE |

Larger vocabularies encode text more efficiently (fewer tokens per sentence),
which reduces sequence length and allows more text to fit in the context window.

## Run the Tests

```bash
uv run pytest tests/unit/test_tokenizer.py -v
```

`BPETokenizer` tests are automatically skipped until you complete this task.
`CharTokenizer` tests run immediately and should all pass.
