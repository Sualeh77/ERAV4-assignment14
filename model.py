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

class DeepSeekV3(nn.Module):
    """
    DeepSeek V3-style LLaMA decoder-only language model.

    Usage:
        cfg = DeepSeekConfig()
        model = DeepSeekV3(cfg)

        input_ids: LongTensor (B, T)
        logits = model(input_ids)
    """
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        ) # (Vocab_Size, Hidden_Size) (49152 x 768)

        self.layers = nn.ModuleList(
            [DeepSeekBlock(config) for _ in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        ) # (Hidden_Size, Vocab_Size) (768 x 49152)

        # tie weights
        self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for Linear and Embedding layers.
        For Linear layers with NANDeepSeek_SCALE_INIT attribute, scale std by sqrt(2 * num_layers).
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANDeepSeek_SCALE_INIT'):
                # Initialize marked linear layers using formula: std = 0.02 * sqrt(2 * num_layers) this 2 x number of layers is because of each block has 2 residual connection. So it actually based on number of residual connection in the model.
                std *= (2 * self.config.num_hidden_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 1 / math.sqrt(module.weight.shape[1])) # std should be calculated using embedding vector size with formula: std = 1 / sqrt(embedding_vector_size)