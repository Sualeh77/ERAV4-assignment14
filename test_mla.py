
import torch
from model import DeepSeekConfig, MultiHeadLatentAttention, build_rope_cache

def test_mla_shapes():
    print("Testing MultiHeadLatentAttention shapes...")
    config = DeepSeekConfig()
    # Config: hidden=768, heads=12, kv_heads=4, head_dim=64
    # Expected internal dims: n_rope=32, n_nope=32
    
    model = MultiHeadLatentAttention(config)
    
    B, T = 2, 10
    x = torch.randn(B, T, config.hidden_size)
    
    # RoPE cache
    # DeepSeekV3 builds cache with full head_dim (64) -> cos has dim 32
    cos, sin = build_rope_cache(T, config.head_dim, config.rope_theta, x.device, x.dtype)
    
    print(f"Input shape: {x.shape}")
    print(f"Cos shape: {cos.shape}")
    
    # Forward
    out, cache = model(x, cos, sin)
    
    print(f"Output shape: {out.shape}")
    
    if out.shape == (B, T, config.hidden_size):
        print("SUCCESS: Output shape matches expected.")
    else:
        print(f"FAILURE: Output shape {out.shape} != {(B, T, config.hidden_size)}")

    # Check Cache
    if cache is not None:
        # cache is (k, v)
        k, v = cache
        # Expected K shape: (B, kv_heads, T, head_dim) = (2, 4, 10, 64)
        print(f"Cache K shape: {k.shape}")
        if k.shape == (B, config.num_key_value_heads, T, config.head_dim):
             print("SUCCESS: Cache K shape matches expected.")
        else:
             print(f"FAILURE: Cache K shape {k.shape} != {(B, config.num_key_value_heads, T, config.head_dim)}")

if __name__ == "__main__":
    test_mla_shapes()
