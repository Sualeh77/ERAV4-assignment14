---
title: Deepseek V3 Trained On Tinyshakespear For Fun
emoji: ⚡
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: false
short_description: Deepseek V3 architecture build and trained from scratch
---

# DeepSeek V3 Architecture Implementation

This repository contains a PyTorch implementation of the DeepSeek V3 architecture, trained on the TinyShakespeare dataset. The implementation follows the architectural details from the DeepSeek V3 technical report, focusing on Multi-Head Latent Attention (MLA) and Mixture-of-Experts (MoE) with auxiliary-loss-free load balancing.

## Key Implementation Features

### 1. Multi-Head Latent Attention (MLA)
We implemented the standard MLA with Decoupled RoPE strategy:
-   **Latent Compression**: Both Key-Value (KV) and Query (Q) are compressed into a latent vector (`latent_dim`), reducing KV cache usage.
-   **Decoupled RoPE**:
    -   Q and K are split into 'pe' (positional) and 'nope' (non-positional/content) parts.
    -   RoPE is applied *only* to the 'pe' part.
    -   The 'nope' part is projected directly from the latent representation.
-   **GQA (Group Query Attention)**: The model uses 12 attention heads and 4 KV heads. K and V are expanded to match the number of Q heads during attention.

### 2. Mixture-of-Experts (MoE)
The model replaces standard Feed-Forward Networks (FFNs) with MoE layers:
-   **Shared + Routed Experts**: We use a hybrid approach with `num_shared_experts=1` (always active) and `num_routed_experts=7`.
-   **Top-K Routing**: For each token, the top-2 experts are selected from the routed experts.
-   **Loss-less Load Balancing**: Instead of using an auxiliary loss term, we implement the DeepSeek V3 strategy of updating bias terms dynamically based on expert load. This ensures balanced usage of experts without interfering with the primary training objective.

## Training Details

The model was trained on the TinyShakespeare dataset in two phases to handle the training resumption and learning rate scheduling effectively.

### Training Phases
1.  **Phase 1 (Steps 0 - 2500)**: Initial training phase.
2.  **Phase 2 (Steps 2500 - 10000)**: Resumed training with a `ResumableDataLoader` to continue exactly where Phase 1 left off.

### Results
-   **Minimum Loss**: Achieved a loss of **~0.01** at step **10002**.
-   **Dataset**: TinyShakespeare
-   **Architecture Config**:
    -   Layers: 8
    -   Hidden Size: 768
    -   Heads: 12
    -   KV Heads: 4
    -   Experts: 8 (1 Shared + 7 Routed)

## Code Structure
-   `model.py`: Complete implementation of DeepSeek V3, including `MultiHeadLatentAttention` and `DeepSeekMoE` classes.
-   `train.py`: Training script using PyTorch Lightning, with custom `ResumableDataLoader` and `WarmupStableDecayLR` scheduler.
