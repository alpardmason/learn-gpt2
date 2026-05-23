# Guide 14 — Further Reading

Congratulations on completing the reimplementation. Below is a curated reading
list to go deeper on each component.

---

## Foundational Papers

| Paper | What to read it for |
|-------|---------------------|
| Vaswani et al. (2017) — *Attention Is All You Need* | Original Transformer architecture; post-norm residuals, sinusoidal positions |
| Radford et al. (2019) — *GPT-2* (`references/language_models_are_unsupervised_multitask_learners.pdf`) | Architecture you just implemented |
| Ba et al. (2016) — *Layer Normalization* | Mathematical derivation of LayerNorm |
| Hendrycks & Gimpel (2016) — *GELU* (`references/gelu_paper.pdf`) | GELU derivation and comparison to ReLU |
| Sennrich et al. (2016) — *BPE for NMT* | Original BPE paper |
| Press & Wolf (2017) — *Weight Tying* | Justification for tying embeddings and lm\_head |
| Loshchilov & Hutter (2019) — *AdamW* | Why decouple weight decay from adaptive moments |

---

## Architecture Improvements (Post GPT-2)

| Paper | Contribution |
|-------|-------------|
| Brown et al. (2020) — *GPT-3* | Scale; few-shot prompting emerges |
| Su et al. (2022) — *RoFormer / RoPE* | Rotary positional embeddings |
| Shazeer (2020) — *GLU Variants* | SwiGLU activation |
| Zhang & Sennrich (2019) — *Root Mean Square Layer Norm* | RMSNorm |
| Ainslie et al. (2023) — *GQA* | Grouped-query attention |
| Dao et al. (2022) — *FlashAttention* | Memory-efficient exact attention |
| Touvron et al. (2023) — *LLaMA 2* | Combines RoPE + SwiGLU + GQA + RMSNorm |

---

## Training at Scale

| Paper | What to read it for |
|-------|---------------------|
| Kaplan et al. (2020) — *Neural Scaling Laws* | How loss scales with compute, data, and parameters |
| Hoffmann et al. (2022) — *Chinchilla* | Optimal compute-tokens tradeoff; revised scaling laws |
| Chowdhery et al. (2022) — *PaLM* | Training recipe for very large models |

---

## Implementation References

| Resource | Notes |
|----------|-------|
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | Line-by-line walkthrough of the original Transformer |
| [Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 2-hour video building a GPT from scratch |
| [Karpathy: nanoGPT](https://github.com/karpathy/nanoGPT) | Clean minimal PyTorch GPT-2 training code |
| `references/nanochat/` | Modern full-stack GPT training (RoPE, GQA, SwiGLU) |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | Production GPT-2 (`modeling_gpt2.py`) |

---

## Next Projects

Now that you've implemented GPT-2, natural next steps:

1. **MLX reimplementation** — same architecture, Apple Silicon, lazy evaluation
2. **JAX + Flax reimplementation** — functional programming, XLA compilation
3. **Add KV cache** — implement cached autoregressive inference and measure speedup
4. **Fine-tune on a small dataset** — understand instruction tuning
5. **Implement FlashAttention** — understand tiled attention and CUDA memory layout
6. **Reproduce Chinchilla** — run scaling law experiments on tiny models

---

## Glossary

| Term | Definition |
|------|-----------|
| BPE | Byte-Pair Encoding — tokenisation algorithm |
| Pre-norm | LayerNorm applied before the sub-layer (GPT-2 style) |
| Post-norm | LayerNorm applied after the residual add (original Transformer) |
| RoPE | Rotary Position Embedding — encodes relative positions in Q and K |
| GQA | Grouped-Query Attention — K/V have fewer heads than Q |
| MHA | Multi-Head Attention — full attention (GPT-2 uses this) |
| KV cache | Cached K and V tensors to speed up autoregressive generation |
| Weight tying | Sharing `wte.weight` with the output projection |
| Residual stream | The tensor `x` that flows through all blocks unchanged except for residual adds |
| Perplexity | $\exp(\text{loss})$ — common LM evaluation metric |
