"""This script is made to preprocess the JSB dataset by converting it in three .pt files, containing training, vallidation and test data"""

import json
import torch
from pathlib import Path
import os

def load_jsb_dataset(json_path):
    """Load the pre-split dataset into a dictionary of train/val/test lists."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Each split contains a list of sequences (each sequence = list of 4-note chords)
    return {
        "train": data["train"],
        "valid": data["valid"],
        "test": data["test"]
    }

def convert_to_tensor(sequences):
    """
    Convert a list of sequences into a list of tensors.
    Each sequence: list of 4-note chords (list[int]) → Tensor[seq_len, 4]
    """
    tensor_seqs = []
    for seq in sequences:
        tensor = torch.tensor(seq, dtype=torch.long)  # Shape: [seq_len, 4]
        tensor_seqs.append(tensor)
    return tensor_seqs

def save_tensor_data(tensor_seqs, out_path):
    """
    Save the tensor list to disk.
    Each item is a tensor of shape [seq_len, 4], representing one SATB sequence.
    """
    torch.save(tensor_seqs, out_path)
    print(f"Saved: {out_path} ({len(tensor_seqs)} sequences)")

def main():
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = Path(os.path.join(project_path, "data", "raw", "Jsb16thSeparated.json"))
    output_dir = Path(os.path.join(project_path, "data", "processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_jsb_dataset(input_path)

    for split in ["train", "valid", "test"]:
        tensor_seqs = convert_to_tensor(dataset[split])
        save_tensor_data(tensor_seqs, output_dir / f"{split}.pt")

    print("🎉 All splits processed and saved.")

if __name__ == "__main__":
    main()