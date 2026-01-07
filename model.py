"""
DeepSeek V3 Architecture Implementation

This module implements the DeepSeek V3 architecture from scratch.
The implementation is based on the SmolLM2-135M model, which is fundamentally
a Llama decoder-only architecture.

Key Features to be implemented:
1. Multi-Head Latent Attention (MHLA): 
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

# =========================
# 1. Config
# =========================
@dataclass
class DeepSeekConfig:
    """Configuration for the DeepSeek V3 model."""
    vocab_size: int = 49152          # from HF config
    hidden_size: int = 768           # "hidden_size"
    intermediate_size: int = 1536    # "intermediate_size"
    num_hidden_layers: int = 8      # "num_hidden_layers" - reduced from 30 to 12 cos model was not fitting in my mac's ram
    num_attention_heads: int = 12     # "num_attention_heads"
    num_key_value_heads: int = 4     # "num_key_value_heads"
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

    compression_ratio: int = 8 # compression ratio (for MHLA)
    num_experts: int = 8 # Total number of experts for Mixture of Experts (MoE)
    num_shared_experts: int = 1 # num_shared_experts : The number of experts that are always active
    top_k_experts: int = 2 # top_k_experts : The number of experts to be selected for each token

# =========================
# 2. RMSNorm
# =========================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    Used in LLaMA / SmolLM2 instead of LayerNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # rms = sqrt(mean(x^2)), but we can use rsqrt for stability
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

# =========================
# 3. RoPE (Rotary Positional Embeddings)
# =========================

def rope_freqs(head_dim: int, base: float, device, dtype):
    """
    Compute inverse frequencies for RoPE.
    """
    half_dim = head_dim // 2
    # Equivalent to: base^{ -2i / d }
    freq_seq = torch.arange(half_dim, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))
    return inv_freq  # shape: (half_dim,)

def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float,
    device,
    dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build cosine and sine caches for RoPE.
    Returns:
        cos: (1, 1, seq_len, head_dim/2)
        sin: (1, 1, seq_len, head_dim/2)
    """
    inv_freq = rope_freqs(head_dim, base, device, dtype)   # (half_dim,)
    # Positions
    t = torch.arange(seq_len, device=device, dtype=dtype)  # (seq_len,)
    freqs = torch.outer(t, inv_freq)                      # (seq_len, half_dim)
    cos = freqs.cos()[None, None, :, :]                   # (1,1,seq_len,half_dim)
    sin = freqs.sin()[None, None, :, :]                   # (1,1,seq_len,half_dim)
    return cos, sin

def apply_rope(
    x: torch.Tensor,  # (B, n_head, T, head_dim)
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to last dimension of x.
    cos, sin are broadcast to match (..., head_dim/2).
    """
    b, h, t, d = x.shape
    half = d // 2

    x1 = x[..., :half] # (B, n_head, T, head_dim/2)
    x2 = x[..., half:] # (B, n_head, T, head_dim/2)

    # cos/sin: (1,1,T,half) -> broadcast over B,h
    cos_t = cos[..., :t, :]
    sin_t = sin[..., :t, :]

    x1_rot = x1 * cos_t - x2 * sin_t
    x2_rot = x1 * sin_t + x2 * cos_t

    return torch.cat([x1_rot, x2_rot], dim=-1) # (B, n_head, T, head_dim)

# =========================
# 4. Attention (MHLA - Multi Head Latent Attention)
# =========================
class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek V3 Multi-Head Latent Attention (MLA) with Group Query Attention (GQA).
    
    Architecture:
    - Latent Compression: KV compression into a latent vector (latent_dim).
    - Decoupled RoPE: 
        - Queries and Keys are split into 'pe' (positional) and 'nope' (non-positional) parts.
        - RoPE is applied ONLY to the 'pe' part.
        - 'nope' part comes from the latent projection.
    - GQA (Group Query Attention):
        - k_heads = v_heads = num_key_value_heads (4).
        - q_heads = num_attention_heads (12).
        - K and V are expanded (repeated) to match Q heads for attention.

    Dimensions Breakdown for this config (hidden=768, heads=12, kv_heads=4):
    - head_dim = 768 / 12 = 64.
    - We split head_dim into:
        - nope_dim = 32 (Content part)
        - rope_dim = 32 (RoPE part)
    
    Q Projections (12 heads):
    - q_d (latent) -> q_nope (12 * 32 = 384)
    - q_d (latent) -> q_rope (12 * 32 = 384)
    - Total Q dim = 384 + 384 = 768.

    K Projections (4 heads - GQA):
    - k_proj_u (latent) -> k_nope (4 * 32 = 128)  | 128 needs 3 repetation to reach 384
    - rope_k (input)    -> k_rope (4 * 32 = 128)  | 128 needs 3 repetation to reach 384
    - Total K dim = 128 + 128 = 256. | 256 needs 3 repetation to reach 768

    V Projection (4 heads):
    - v_proj_u (latent) -> v (4 * 64 = 256) (No RoPE split for V) | 256 needs 3 repetation to reach 768
    """
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads # 12
        self.n_kv_heads = config.num_key_value_heads # 4
        self.head_dim = config.head_dim # 64
        self.hidden_size = config.hidden_size # 768
        
        # Split head dimension for Decoupled RoPE (DeepSeek Strategy)
        # Typically split 50/50 or similar. With head_dim=64, we use 32/32.
        self.nope_head_dim = self.head_dim // 2 # 32
        self.rope_head_dim = self.head_dim // 2 # 32

        self.latent_dim = self.hidden_size // config.compression_ratio # 768 // 8 = 96

        assert self.hidden_size == self.n_heads * self.head_dim

        # 1. Down-Projections (Compression)
        # ---------------------------------
        # KV Compression: Project Input -> Latent
        self.kv_proj_d = nn.Linear(
            self.hidden_size, # 768
            self.latent_dim, # 96
            bias=config.attention_bias,
        )
        # Q Compression: Project Input -> Latent
        self.q_proj_d = nn.Linear(
            self.hidden_size, # 768
            self.latent_dim, # 96
            bias=config.attention_bias,
        )

        # 2. Up-Projections (Decompression / Head Generation)
        # ---------------------------------------------------
        
        # Query Heads (12 heads)
        # Q Content (Nope): Latent -> 12 * 32 = 384
        self.q_proj_u = nn.Linear(
            self.latent_dim, # 96
            self.n_heads * self.nope_head_dim, # 384
            bias=config.attention_bias,
        )
        # Q RoPE (Pe): Latent -> 12 * 32 = 384
        self.rope_q = nn.Linear(
            self.latent_dim, # 96
            self.n_heads * self.rope_head_dim, # 384
            bias=config.attention_bias,
        )

        # Key Heads (4 heads - GQA)
        # K Content (Nope): Latent -> 4 * 32 = 128
        # NOTE: Repeating 3 times for GQA would yield 384 which will become 768 after adding rope.
        self.k_proj_u = nn.Linear(
            self.latent_dim, # 96
            self.n_kv_heads * self.nope_head_dim, # 128
            bias=config.attention_bias,
        )
        # K RoPE (Pe): Input -> 4 * 32 = 128
        # NOTE: Takes original Input 'x', not latent. DeepSeek uses decoupled strategy.
        # TODO: Verify if this approach of generating rope embedding of 128 dim and then reapeating it 3 times to make 384 Or
        #       generating rope embedding of 128 dim and then adding it with 128 dim K which yield 256 dim vector, then repeating it 3 times to make 768.
        #       is this correct or we should generate rope embedding of 384 dim and then add it with 3 times repeated k vector of 384 dim which yield 768 dim vector.
        # ANSWER TO TODO: The correct and most efficient approach is to generating RoPE for the keys (4 heads)
        #                 and combine them, THEN expand to 12 heads.
        #                 Logic: 
        #                   1. Use 4 heads for projection (Save params/compute).
        #                   2. Apply RoPE to 4 heads (Save compute).
        #                   3. Concatenate (yields 256 dim = 4 heads * 64).
        #                   4. Only then expand to 12 heads (GQA) for attention.
        #                 Doing it the other way (expanding valid RoPE 128->384 first) would waste compute.
        self.rope_k = nn.Linear(
            self.hidden_size, # 768
            self.n_kv_heads * self.rope_head_dim, # 128
            bias=config.attention_bias,
        )

        # Value Heads (4 heads - GQA)
        # V: Latent -> 4 * 64 = 256
        self.v_proj_u = nn.Linear(
            self.latent_dim, # 96
            self.n_kv_heads * self.head_dim, # 256
            bias=config.attention_bias,
        )

        # Output Projection
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, # 768
            self.hidden_size, # 768
            bias=config.attention_bias,
        )
        self.o_proj.NANDeepSeek_SCALE_INIT = True  # mark for scaled initialization

    def forward(
        self,
        x: torch.Tensor,                # (B, T, C) or (B, 1, C) for inference
        cos: torch.Tensor,              # (1,1,T,head_dim/2) or (1,1,1,head_dim/2) for inference
        sin: torch.Tensor,              # (1,1,T,head_dim/2) or (1,1,1,head_dim/2) for inference
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) or (B,1,1,T)
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (k_cache, v_cache)
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        # ==========================================================
        # 1. Projections & Latent Compression
        # ==========================================================
        
        # A. KV Compression
        kv_d = self.kv_proj_d(x) # (B,T,C) -> (B,T,latent_dim) | (8, 2048, 768) -> (8, 2048, 96)

        # B. Q Compression
        q_d = self.q_proj_d(x) # (B,T,C) -> (B,T,latent_dim) | (8, 2048, 768) -> (8, 2048, 96)

        # ==========================================================
        # 2. Generate Head Components
        # ==========================================================

        # Queries (12 Heads)
        # ------------------
        # Q Content (Nope): (B, T, 384) -> Reshape -> (B, T, 12, 32)
        q_nope = self.q_proj_u(q_d).view(B, T, self.n_heads, self.nope_head_dim) # (B,T,latent_dim) -> (B,T,12*32) - (B,T,384) -> (B,T,12,32)
        
        # Q RoPE (Pe): (B, T, 384) -> Reshape -> (B, T, 12, 32)
        q_pe = self.rope_q(q_d).view(B, T, self.n_heads, self.rope_head_dim) # (B,T,latent_dim) -> (B,T,12*32) - (B,T,384) -> (B,T,12,32)

        # Keys (4 Heads)
        # --------------
        # K Content (Nope): (B, T, 128) -> Reshape -> (B, T, 4, 32)
        k_nope = self.k_proj_u(kv_d).view(B, T, self.n_kv_heads, self.nope_head_dim)

        # K RoPE (Pe): (B, T, 128) -> Reshape -> (B, T, 4, 32)
        # Generated from original input x
        k_pe = self.rope_k(x).view(B, T, self.n_kv_heads, self.rope_head_dim) # (B,T,hidden_size) -> (B,T,4*32) - (B,T,128) -> (B,T,4,32)

        # Values (4 Heads)
        # ----------------
        # V: (B, T, 256) -> Reshape -> (B, T, 4, 64)
        v = self.v_proj_u(kv_d).view(B, T, self.n_kv_heads, self.head_dim) # (B,T,latent_dim) -> (B,T,4*64) - (B,T,256) -> (B,T,4,64)

        # ==========================================================
        # 3. Apply RoPE (Decoupled Strategy)
        # ==========================================================
        # We only apply RoPE to the 'pe' parts (q_pe, k_pe).
        # Important: q_pe is 32-dim. standard RoPE pairs 2 values.
        # We need RoPE frequencies for 32/2 = 16 pairs.
        # The passed cos/sin are for head_dim=64 (32 pairs).
        # We must slice cos/sin to use only the first 16 pairs.
        
        cos_sliced = cos[..., :self.rope_head_dim//2] # (..., 16)
        sin_sliced = sin[..., :self.rope_head_dim//2] # (..., 16)

        # Apply RoPE
        # q_pe: (B, T, 12, 32) -> transpose -> (B, 12, T, 32)
        q_pe_rot = apply_rope(q_pe.transpose(1, 2), cos_sliced, sin_sliced).transpose(1, 2)
        
        # k_pe: (B, T, 4, 32) -> transpose -> (B, 4, T, 32)
        k_pe_rot = apply_rope(k_pe.transpose(1, 2), cos_sliced, sin_sliced).transpose(1, 2)

        # ==========================================================
        # 4. Concatenate Heads (Nope + Pe)
        # ==========================================================
        
        # Q = [q_nope, q_pe_rot] -> (B, T, 12, 32+32) = (B, T, 12, 64)
        q = torch.cat([q_nope, q_pe_rot], dim=-1) # (B, T, 12, 64)
        
        # K = [k_nope, k_pe_rot] -> (B, T, 4, 32+32) = (B, T, 4, 64)
        k = torch.cat([k_nope, k_pe_rot], dim=-1) # (B, T, 4, 64)

        # Transpose for Attention: (B, T, h, d) -> (B, h, T, d)
        q = q.transpose(1, 2) # (B, 12, T, 64)
        k = k.transpose(1, 2) # (B, 4, T, 64)
        v = v.transpose(1, 2) # (B, 4, T, 64)

        # ==========================================================
        # 5. KV Cache & GQA Expansion
        # ==========================================================
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            # past_k, past_v: (B, n_kv_heads, past_len, head_dim)
            k = torch.cat([past_k, k], dim=2)  # (B, n_kv_heads, past_len + T, head_dim)
            v = torch.cat([past_v, v], dim=2)  # (B, n_kv_heads, past_len + T, head_dim)
            seq_len = k.shape[2]
        else:
            seq_len = T

        # Update cache (Pre-expansion)
        present_key_value = (k, v) if use_cache else None

        # GQA: Expand K/V to match Q heads (12)
        # We have 4 KV heads, 12 Q heads. Factor = 3.
        if self.n_kv_heads != self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1) # (B, 12, seq_len, 64)
            v = v.repeat_interleave(repeat_factor, dim=1) # (B, 12, seq_len, 64)

        # ==========================================================
        # 6. Attention
        # ==========================================================
        
        if past_key_value is None:
            # Full sequence: use causal mask
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        else:
            # With KV cache: no causal mask needed (already handled by cache structure)
            # But we still need to handle attention_mask if provided
            attn_mask = None
            if attention_mask is not None:
                # Convert attention_mask to the right format for Flash Attention
                if attention_mask.dim() == 2:
                    attn_mask = attention_mask[:, None, None, :]
                attn_mask = attn_mask.expand(B, self.n_heads, T, seq_len)

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=False,  # Already handled by KV cache
                scale=1.0 / math.sqrt(self.head_dim),
            )

        # Reshape back: (B, h, T, d) -> (B, T, h, d) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        out = self.o_proj(out)

        return out, present_key_value
        


# =========================
# 5. MOE -> MLP (SwiGLU), MOE, Auxilary Lossless Load Balancing
# =========================
class DeepSeekExpertLayer(nn.Module):
    """
    SwiGLU MLP:
        z = W1(x) -> split -> (x1, x2)
        out = W2( SiLU(x1) * x2 )
    """
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,   # for SwiGLU split (2 x 1536 = 3072)
            bias=config.mlp_bias,
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,   # 1536
            config.hidden_size,   # 768
            bias=config.mlp_bias,
        )
        self.down_proj.NANDeepSeek_SCALE_INIT = True     # mark for scaled initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)# (B,T,C) -> (B,T,2*intermediate_size) -> (B,T,1536*2) -> (B,T,3072)
        gate_proj, up_proj = x.chunk(2, dim=-1)  # (B,T,2*intermediate_size) = (B,T,3072) -> (B,T,intermediate), (B,T,intermediate) = (B,T,1536), (B,T,1536)
        return self.down_proj(F.silu(gate_proj) * up_proj) # (B,T,intermediate) * (B,T,intermediate) -> (B,T,intermediate) -> (B,T,hidden_size) = (B,T,768)


class DeepSeekMoE(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_experts = config.num_experts # 8
        self.num_shared_experts = config.num_shared_experts # 1
        self.num_routed_experts = self.num_experts - self.num_shared_experts # 7
        self.top_k_experts = config.top_k_experts # 2
        self.hidden_size = config.hidden_size # 768

        # Shared Experts
        self.shared_experts = nn.ModuleList([DeepSeekExpertLayer(config) for _ in range(self.num_shared_experts)])

        # Routed Experts
        self.routed_experts = nn.ModuleList([DeepSeekExpertLayer(config) for _ in range(self.num_routed_experts)])

        # Router Components
        self.router = nn.Linear(self.hidden_size, self.num_routed_experts)
        self.routing_bias = nn.Parameter(torch.zeros(self.num_routed_experts))
        self.last_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Process through shared experts | (B,T,C) -> (B,T,C)
        shared_output = sum(expert(x) for expert in self.shared_experts) / self.num_shared_experts # Average if multiple shared experts

        # Calculate routing scores | (B,T,C) -> (B,T,num_routed_experts) (B,T,7)
        routing_logits = self.router(x) + self.routing_bias # (B,T,7)

        # Get Top-K Experts per token | (B,T,7) -> (B,T,2)
        routing_probs = torch.sigmoid(routing_logits) # (B,T,7)
        scores, indices = torch.topk(routing_probs, k=self.top_k_experts, dim=-1) # ((B,T,2), (B,T,2))
        
        if self.training:
            self.last_indices = indices.detach()
        
        # Process through selected experts | (B,T,C) -> (B,T,C)
        combined_output = torch.zeros_like(x) # (B,T,C)
        for k in range(self.top_k_experts):
            # This means at a time for each selected expert we are picking it's index and score for every token in the batch
            expert_indices = indices[..., k] # (B,T)
            expert_scores = scores[..., k:k+1] # (B,T)

            # Process each expert
            for i in range(self.num_routed_experts):
                mask = (expert_indices == i) # This will create a boolean mask (True for selected expert, False otherwise) of shape (B,T) where the expert index matches
                if mask.any():
                    expert_input = x[mask] # (B,T,C) -> (B,N,C) Where N is the number of tokens for which the expert is selected
                    expert_output = self.routed_experts[i](expert_input) # (B,N,C) -> (B,N,C)
                    combined_output[mask] += expert_output * expert_scores[mask] # (B,N,C) -> (B,T,C)
                    # Since combined_output is of size (B,T,C) and we are adding (B,N,C) by accessing (B,N,C) from combined_output, Eventually we get output of size (B,T,C)

        final_output = shared_output + combined_output # (B,T,C)
        return final_output

    def update_bias_terms(self, expert_load):
        # Adjust bias terms based on expert load
        target_load = 1.0 / self.num_routed_experts
        load_diff = expert_load - target_load

        # Dynamic update rate based on the magnitude of the load imbalance
        update_rate = 0.1 * torch.abs(load_diff)
        
        self.routing_bias.data -= update_rate * load_diff

class DeepSeekMLP(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.moe = DeepSeekMoE(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe(x)

# =========================
# 6. Transformer Block
# =========================
class DeepSeekBlock(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MultiHeadLatentAttention(config) # Multi-Head Latent Attention (MHLA / MLA)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = DeepSeekMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm + residual for attention
        attn_out, present_key_value = self.attn(
            self.attn_norm(x), cos, sin, attention_mask, past_key_value, use_cache
        )
        x = x + attn_out

        # Pre-norm + residual for MLP
        x = x + self.mlp(self.mlp_norm(x))
        return x, present_key_value        

# =============================================
# 7. DeepSeek V3 Model Architecture
#  DeepSeek V3 follows the LLaMA-style decoder-only Transformer architecture.
# =============================================
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

    def forward(
        self,
        input_ids: torch.Tensor,            # (B, T)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = input_ids.shape

        # For inference with KV cache, we might have T=1
        if past_key_values is None:
            assert T <= self.config.max_position_embeddings, (
                f"Sequence length {T} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )
            seq_len = T
        else:
            # With KV cache, current sequence length is past_len + T
            past_len = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            seq_len = past_len + T
            assert seq_len <= self.config.max_position_embeddings, (
                f"Total sequence length {seq_len} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        # Embedding
        x = self.embed_tokens(input_ids)  # (B,T) -> (B,T,C)

        # RoPE cache - build for the full sequence length (past + current)
        cos, sin = build_rope_cache(
            seq_len=seq_len,
            head_dim=self.config.head_dim,
            base=self.config.rope_theta,
            device=x.device,
            dtype=x.dtype,
        )

        # If using KV cache, we only need cos/sin for current positions
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            # Slice to get only the current positions for RoPE
            cos = cos[..., past_len:, :]
            sin = sin[..., past_len:, :]

        # Layers
        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(x, cos, sin, attention_mask, past_kv, use_cache)
            if use_cache:
                present_key_values.append(present_kv)

        # Final norm + lm head
        x = self.norm(x)
        logits = self.lm_head(x)  # (B,T,C) -> (B,T,vocab_size)
        return logits, present_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text using KV cache for efficient inference.
        
        Args:
            input_ids: (B, T) input token ids
            max_new_tokens: maximum number of new tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling (keep top k tokens)
            top_p: nucleus sampling (keep tokens with cumulative probability <= top_p)
            eos_token_id: end-of-sequence token id (stop generation when encountered)
        
        Returns:
            generated_ids: (B, T + max_new_tokens) generated token ids
        """
        self.eval()
        device = input_ids.device
        B, T = input_ids.shape

        # Start with input_ids
        generated_ids = input_ids.clone()
        past_key_values = None

        for step in range(max_new_tokens):
            # Forward pass with KV cache
            # On first iteration, use full input_ids. On subsequent iterations, use only last token
            if past_key_values is None:
                # First iteration: process full sequence
                current_input = generated_ids
            else:
                # Subsequent iterations: only process the last generated token
                current_input = generated_ids[:, -1:]
            
            logits, past_key_values = self.forward(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Get logits for the last token (always the last position in logits)
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated_ids

# =========================
# 8. Quick self-test
# =========================
if __name__ == "__main__":
    # Tiny sanity check: runs a forward pass on random input
    cfg = DeepSeekConfig()
    model = DeepSeekV3(cfg)

    B, T = 2, 16
    x = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits, _ = model(x)

    print("Input shape :", x.shape)
    print("Logits shape:", logits.shape)  # should be (2, 16, vocab_size)
