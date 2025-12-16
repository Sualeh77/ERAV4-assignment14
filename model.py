"""
DeepSeek V3 Architecture Implementation

This module implements the DeepSeek V3 architecture from scratch.
The implementation is based on the SmolLM2-135M model, which is fundamentally
a Llama decoder-only architecture.

Key Features to be implemented:
1. Multi-Head Latent Attention (MLHA): 
   - Known as MLA (Multi-Head Latent Attention) in DeepSeek papers.
   - Optimizes KV cache usage and inference efficiency.

2. Mixture-of-Experts (MoE) with Loss-less Load Balancing:
   - Replaces standard FFN layers with MoE layers.
   - Uses auxiliary-loss-free or loss-less load balancing strategies 
     specifically for DeepSeek V3/V2.

Reference:
- Base Architecture: SmolLM2-135M (Llama-based)
- Target Architecture: DeepSeek V3
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DeepSeekConfig:
    """Configuration for the DeepSeek V3 model."""
    vocab_size: int = 49152          # from HF config
    hidden_size: int = 768           # "hidden_size"
    intermediate_size: int = 1536    # "intermediate_size"
    num_hidden_layers: int = 30      # "num_hidden_layers"
    num_attention_heads: int = 12     # "num_attention_heads"
    max_position_embeddings: int = 2048  # "max_position_embeddings" - Max sequence length

    # Positional / RoPE
    rope_theta: float = 100000.0     # "rope_theta"

    # Norm / numerical
    rms_norm_eps: float = 1e-5       # "rms_norm_eps"

    # Biases
    attention_bias: bool = False     # "attention_bias"
    mlp_bias: bool = False           # "mlp_bias"

    # Misc
    dtype: torch.dtype = torch.bfloat16

    @property
    def head_dim(self) -> int:
        # Keeping per head dimension as 64 for DeepSeek V3 (768 / 12).
        return self.hidden_size // self.num_attention_heads # 768 / 12 = 64

    compression_ratio: int = 8 # compression ratio (for MLHA)
    num_experts: int = 8 # Total number of experts for Mixture of Experts (MoE)
    num_shared_experts: int = 1 # num_shared_experts : The number of experts that are always active
    top_k_experts: int = 2 # top_k_experts : The number of experts to be selected for each token