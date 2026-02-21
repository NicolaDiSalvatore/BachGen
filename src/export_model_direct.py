import glob
import os

import numpy as np
import torch

torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])

def export_best_model_direct(experiments_dir="experiments/transformer_bach_dataset", output_path="deploy/model.pth"):
    """
    Exports the best model by scanning local checkpoint files directly, bypassing MLflow DB.
    """
    print(f"Scanning for checkpoints in {experiments_dir}...")


    search_pattern = os.path.join(experiments_dir, "**", "best_val.pth")
    checkpoints = glob.glob(search_pattern, recursive=True)

    if not checkpoints:
        print("No 'best_val.pth' checkpoints found.")
        return False

    print(f"Found {len(checkpoints)} checkpoints.")

    best_checkpoint_path = None
    best_val_loss = float('inf')

    for cp_path in checkpoints:
        try:
            checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)

            if 'avg_validation_loss' in checkpoint:
                val_loss = checkpoint['avg_validation_loss']
                print(f"Checked {os.path.basename(os.path.dirname(os.path.dirname(cp_path)))}: val_loss={val_loss}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_path = cp_path
            else:
                 print(f"Warning: 'avg_validation_loss' not found in {cp_path}")

        except Exception as e:
            print(f"Could not load {cp_path}: {e}")
            continue

    if best_checkpoint_path:
        print(f"\nBest checkpoint found: {best_checkpoint_path}")
        print(f"Validation Loss: {best_val_loss}")


        try:
            checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)

            export_data = {
                'model_state_dict': checkpoint['model_state_dict'],
                'config': {
                    'attention_hidden_size': checkpoint.get('attention_hidden_size'),
                    'feedforward_hidden_dim': checkpoint.get('feedforward_hidden_dim'),
                    'num_decoder_layers': checkpoint.get('num_decoder_layers'),
                    'num_attention_heads': checkpoint.get('num_attention_heads'),
                    'embed_dropout': 0.0,
                    'ffn_dropout': 0.0,
                    'attn_dropout': 0.0,
                    'attn_proj_dropout': 0.0,
                    'vocab_size': 92,
                    'seq_len': 2048
                }
            }

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(export_data, output_path)
            print(f"Successfully exported model to {output_path}")
            return True

        except Exception as e:
            print(f"Error saving exported model: {e}")
            return False
    else:
        print("No valid checkpoints found with 'avg_validation_loss'.")
        return False

if __name__ == "__main__":
    export_best_model_direct()
