"""
GPT-2 reimplementation — educational PyTorch version.

Modules:
  config    — GPT2Config dataclass with model size presets
  tokenizer — CharTokenizer (for tests) and BPETokenizer stub
  model     — GELU, LayerNorm, CausalSelfAttention, MLP, Block, GPT2Model
  train     — cross_entropy_loss, cosine_lr_schedule, Trainer
  generate  — top_k_filter, top_p_filter, generate
"""

from gpt2.config import GPT2Config as GPT2Config
from gpt2.model import GPT2Model as GPT2Model
