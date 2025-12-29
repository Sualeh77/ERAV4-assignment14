"""
Training script for DeepSeekV3 using PyTorch Lightning.

Since training it on local on a tiny-shakespear dataset just for fun, Using Training strategy from SmolLM2 paper:
- AdamW optimizer with (β1, β2) = (0.9, 0.95)
- Warmup Stable Decay (WSD) learning rate schedule:
  - 2,000-step warmup phase
  - Peak learning rate: 5.0 × 10^-4 (stable phase)
  - Decay phase: reduce LR to zero over 10% of total training steps
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import AutoTokenizer, AutoConfig

from model import DeepSeekV3, DeepSeekConfig

# Setup logging
def setup_logging(log_dir: Path):
    """Setup text file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__), log_file

class TextDataset(Dataset):
    """Dataset for text data."""
    def __init__(self, text_file: str, tokenizer, block_size: int = 512):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Read and tokenize text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        self.data = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

class WarmupStableDecayLR(L.Callback):
    """
    Warmup Stable Decay (WSD) learning rate schedule.
    - Warmup: 2000 steps in paper, Since only training for 5000 steps, we will use 20% of total steps as warmup steps (1000 steps)
    - Stable: maintain peak LR
    - Decay: reduce to zero over 10% of total steps
    """
    def __init__(self, warmup_steps: int = 2000, peak_lr: float = 5e-4, total_steps: int = 5000):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.decay_steps = int(0.1 * total_steps)  # 10% of total steps
        self.stable_steps = total_steps - warmup_steps - self.decay_steps
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        current_step = trainer.global_step
        
        if current_step < self.warmup_steps:
            # Warmup phase: linear increase
            lr = self.peak_lr * (current_step / self.warmup_steps)
        elif current_step < self.warmup_steps + self.stable_steps:
            # Stable phase: maintain peak LR
            lr = self.peak_lr
        else:
            # Decay phase: linear decrease to zero
            decay_start = self.warmup_steps + self.stable_steps
            decay_progress = (current_step - decay_start) / self.decay_steps
            lr = self.peak_lr * (1.0 - decay_progress)
        
        # Update learning rate
        optimizer = pl_module.optimizers()
        if isinstance(optimizer, torch.optim.Optimizer):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # If it's a list or other structure
            for opt in optimizer:
                for param_group in opt.param_groups:
                    param_group['lr'] = lr

class DeepSeekV3Module(L.LightningModule):
    """PyTorch Lightning module for DeepSeekV3 training."""
    def __init__(
        self,
        config: DeepSeekConfig,
        tokenizer,
        block_size: int = 512,
        warmup_steps: int = 2000,
        peak_lr: float = 5e-4,
        total_steps: int = 5000,
        predict_every: int = 500, # Make it 500 for final training.
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.config = config
        self.tokenizer = tokenizer
        self.block_size = block_size # config.max_position_embeddings = 2048
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.predict_every = predict_every

        # Initialize model
        self.model = DeepSeekV3(config)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        self._batch_start_time = None

        # For generation
        self.example_prompt = "First Citizen:"

    def forward(self, input_ids, attention_mask=None):
        logits, present_key_values = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
        return logits

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.perf_counter()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        # Reshape for loss calculation
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # Update MoE biases for load balancing
        self.update_moe_biases()


        # Generate text every predict_every steps
        if (self.global_step + 1) % self.predict_every == 0:
            # Log scalar loss to text log so it shows up with generations
            logger.info(f"Step {self.global_step + 1} | train_loss={loss.item():.4f}")
            self.generate_and_log()
        
        return loss

    def update_moe_biases(self):
        """
        Update MoE routing bias terms based on expert load.
        This implements the auxiliary-loss-free load balancing strategy.
        """
        # Iterate over all layers to find MoE layers
        for layer in self.model.layers:
            if hasattr(layer.mlp, 'moe'):
                moe = layer.mlp.moe
                if moe.last_indices is not None:
                    # moe.last_indices is (B, T, top_k)
                    indices = moe.last_indices
                    
                    # Calculate expert load
                    # We want to know how many times each expert was selected
                    # indices shape: (B, T, k)
                    # We flatten to (B*T*k)
                    flat_indices = indices.view(-1)
                    
                    # Count occurrences of each expert index
                    # expert_load should be of shape (num_routed_experts,)
                    expert_load = torch.bincount(flat_indices, minlength=moe.num_routed_experts).float()
                    
                    # Normalize load
                    total_tokens = flat_indices.numel()
                    if total_tokens > 0:
                        expert_load = expert_load / total_tokens
                    
                    # Update bias terms
                    moe.update_bias_terms(expert_load)


    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Ensure GPU work is finished before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._batch_start_time is None:
            return
        dt = time.perf_counter() - self._batch_start_time  # seconds
        logger.info(f"step {self.global_step + 1} | dt: {dt*1000:.2f} ms")

    def generate_and_log(self):
        """Generate text and log it."""
        self.model.eval()
        with torch.no_grad():
            # Tokenize prompt
            prompt_ids = self.tokenizer.encode(
                self.example_prompt,
                return_tensors='pt',
                add_special_tokens=False
            ).to(self.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Generate
            generated_ids = self.model.generate(
                prompt_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            dt = time.perf_counter() - t0
            new_tokens = generated_ids.shape[1] - prompt_ids.shape[1]
            toks_per_sec = new_tokens / dt if dt > 0 else float("inf")
            logger.info(f"Generation | new_tokens: {new_tokens} | dt: {dt*1000:.2f} ms | tok/s: {toks_per_sec:,.0f}")

            # Decode
            generated_text = self.tokenizer.decode(
                generated_ids[0].cpu().tolist(),
                skip_special_tokens=True
            )

            # Log to console and file
            logger.info(f"\n{'='*80}")
            logger.info(f"Step {self.global_step + 1} - Generated text:")
            logger.info(f"{generated_text}")
            logger.info(f"{'='*80}\n")
        
        self.model.train()

    def configure_optimizers(self):
        """Configure optimizer with AdamW."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.peak_lr,  # Will be adjusted by scheduler
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        
        # WSD scheduler (implemented as callback)
        return optimizer

    def on_train_start(self):
        """Log model summary at training start."""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info("\n" + "="*80)
        logger.info("MODEL SUMMARY")
        logger.info("="*80)
        logger.info(f"Model: DeepSeekV3")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Block size: {self.block_size}")
        logger.info(f"Warmup steps: {self.warmup_steps}")
        logger.info(f"Peak learning rate: {self.peak_lr}")
        logger.info(f"Total training steps: {self.total_steps}")
        logger.info(f"Predict every: {self.predict_every} steps")
        logger.info("="*80 + "\n")

def main():
    # Set matmul precision for faster training
    torch.set_float32_matmul_precision('high')

    # Configuration
    data_file = Path("../data/input.txt").resolve()
    output_dir = Path("./checkpoints")
    log_dir = Path("./logs")
    block_size = 512 # DeepSeekV3 config.max_position_embeddings = 2048
    batch_size = 4
    num_workers = 8
    max_steps = 10  # 10000

    predict_every = 500 # Make it 500 for final training.
    resume_from_checkpoint = None  # Set to checkpoint path to resume, or None for fresh training

    # Training hyperparameters from paper
    warmup_steps = 1000
    peak_lr = 5e-4
    total_steps = max_steps

    # Setup logging
    global logger
    logger, log_file = setup_logging(log_dir)
    logger.info(f"Logging to: {log_file}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M") # Using SmolLM2 tokenizer for testing training.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Allow DeepSeekConfig to be deserialized from Lightning checkpoints when torch.load
    # uses weights_only=True default (torch>=2.6). This is safe because the class
    # is defined locally in this file.
    try:
        torch.serialization.add_safe_globals([DeepSeekConfig])  # type: ignore[attr-defined]
    except Exception:
        # Fallback for torch versions without add_safe_globals; Lightning will still
        # load normally when weights_only=False.
        pass

    # Load config and create model config
    logger.info("Loading model config...")
    config = DeepSeekConfig()

    # Create dataset
    logger.info(f"Loading dataset from: {data_file}")
    dataset = TextDataset(data_file, tokenizer, block_size=block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create Lightning module
    logger.info("Initializing model...")
    model = DeepSeekV3Module(
        config=config,
        tokenizer=tokenizer,
        block_size=block_size,
        warmup_steps=warmup_steps,
        peak_lr=peak_lr,
        total_steps=total_steps,
        predict_every=predict_every,
    )

    # torch.compile it doesn't work on mac or windows can be used on linux (AWS)
    # model = torch.compile(model)

    # Additional callback to ensure checkpoint at final step
    class FinalCheckpointCallback(L.Callback):
        def on_train_end(self, trainer, pl_module):
            # Save final checkpoint
            final_checkpoint_path = output_dir / f"deepseekv3-final-step-{trainer.global_step:05d}.ckpt"
            trainer.save_checkpoint(str(final_checkpoint_path))
            logger.info(f"Final checkpoint saved: {final_checkpoint_path}")

    final_checkpoint_callback = FinalCheckpointCallback()

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='deepseekv3-{step:05d}-{train_loss:.4f}',
        monitor='train_loss',
        save_top_k=3,
        mode='min',
        every_n_train_steps=predict_every,
        save_last=True,
        save_on_train_epoch_end=False,  # Save based on steps, not epochs
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    wsd_scheduler = WarmupStableDecayLR(
        warmup_steps=warmup_steps,
        peak_lr=peak_lr,
        total_steps=total_steps,
    )

    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name='tensorboard',
    )

    # Create trainer
    trainer = L.Trainer(
        max_steps=max_steps,
        callbacks=[checkpoint_callback, lr_monitor, wsd_scheduler, final_checkpoint_callback],
        logger=tb_logger,
        accelerator='auto',
        devices='auto',
        # Set precision depending on device capabilities.
        # bf16-mixed: CUDA; 32-true: others; MPS supports only 32-true.
        precision='bf16-mixed' if torch.cuda.is_available() else '32-true',
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        enable_checkpointing=True,
    )

    # Train
    logger.info("Starting training...")
    if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.fit(model, dataloader, ckpt_path=resume_from_checkpoint)
    else:
        trainer.fit(model, dataloader)
    
    logger.info("Training completed!")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"Last checkpoint: {checkpoint_callback.last_model_path}")


if __name__ == "__main__":
    main()