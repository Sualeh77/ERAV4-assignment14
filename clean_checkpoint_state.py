
import torch
import sys
from pathlib import Path

def clean_checkpoint(ckpt_path):
    print(f"Processing checkpoint: {ckpt_path}")
    try:
        # Load checkpoint
        # using mmap=True to avoid loading entire file to RAM if possible, though torch.load still loads it.
        # But we need to save it back, so we have to load it.
        # Set weights_only=False to allow loading custom classes like DeepSeekConfig
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Check callbacks
        if 'callbacks' in checkpoint:
            print(f"Found callbacks: {list(checkpoint['callbacks'].keys())}")
            
            # Identify ModelCheckpoint keys
            # PyTorch Lightning keys look like "ModelCheckpoint{'monitor': 'train_loss', ...}"
            keys_to_remove = []
            for key in checkpoint['callbacks']:
                if 'ModelCheckpoint' in key:
                    keys_to_remove.append(key)
            
            if keys_to_remove:
                print(f"Removing ModelCheckpoint keys: {keys_to_remove}")
                for key in keys_to_remove:
                    del checkpoint['callbacks'][key]
                
                # Save back
                print("Saving cleaned checkpoint...")
                torch.save(checkpoint, ckpt_path)
                print("Done. ModelCheckpoint state removed.")
            else:
                print("No ModelCheckpoint state found to remove.")
        else:
            print("No 'callbacks' key found in checkpoint.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    ckpt_file = "checkpoints/deepseekv3-step=00500-train_loss=3.3785.ckpt"
    if len(sys.argv) > 1:
        ckpt_file = sys.argv[1]
    
    clean_checkpoint(ckpt_file)
