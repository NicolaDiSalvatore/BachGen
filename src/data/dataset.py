
"""
This scripts redefine the Dataset class in Pytorch to be adapted to Musictransformer. The main modifications are:
1) The __getitem__ method: gives in output also the length of the original sequence
2) The collate_fn function: gives output tensor as [batch, max_sequence_length, 4], 
    where batch is the number of sequences per batch, max_sequence_length is the length of the longest sequence in the batch, 
    and 4 are the SATB voices

"""

from torch.utils.data import Dataset
import json
from pathlib import Path
import torch
from src.data.vocab import encode_pitch
from typing import Tuple
from os.path import dirname, abspath, join
import random

project_path = dirname(dirname(dirname(abspath(__file__))))
print(f"Project path: {project_path}")

def convert_data_to_tensor(data: list, idx: int) -> torch.Tensor:
    # Output dimension: (time_steps * 4)
    time_steps = [torch.tensor(time_step) for time_step in data[idx]]
    return torch.stack(time_steps).reshape(-1)

class BachDataset(Dataset):
    def __init__(self, split: str, data_path: Path = join(project_path, 'data', 'raw', 'Jsb16thSeparated.json'), min_pitch: int = None, augment: bool = False):
        self.split = split
        self.augment = augment

        if split not in ["train", "valid", "test"]:
            raise ValueError(f"Invalid split: {split}")

        with open(data_path, 'r') as file:
            data = json.load(file)

        # Always store raw data
        self.sequences = data[split]
        self.min_pitch_value = min_pitch

        self.min_pitch = None
        self.max_pitch = None
        self.max_seq_len = None
        self.avg_seq_len = None
        self.pitches = set()


    def __len__(self):
        return len(self.sequences)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # The output is a Tuple containing the 'index'-th sequence and its length (number of time steps it contains).
        # The number of time steps is returned here instead of being calculated in the "collate_fn" function, to evitate their calculation for each dataloader call.
        
        sequence = self.sequences[index]
        
        shift = 0
        if self.augment:
             shift = random.randint(-6, 6)
             
        encoded_sequence = []
        for time_step in sequence:
            encoded_time_step = []
            for p in time_step:
                # Apply shift only if it's a valid pitch (not rest/pad if they were integers, but here input is raw pitch)
                # Assuming raw pitches are > 0. REST_PITCH is handled in encode_pitch.
                # However, raw data from JSON might interpret rests as specific value?
                # Looking at vocab.py, REST_PITCH = -1.
                
                p_shifted = p
                if p > 0: # Assuming 0 or -1 are special, and pitches are > 0
                    p_shifted = p + shift
                
                if self.min_pitch_value is not None:
                    encoded_p = encode_pitch(p_shifted, min_pitch=self.min_pitch_value)
                    encoded_time_step.append(encoded_p)
                else:
                    encoded_time_step.append(p_shifted)
            encoded_sequence.append(encoded_time_step)

        # Convert to tensor
        # Reusing the structure of convert_data_to_tensor but for single sequence
        time_steps = [torch.tensor(time_step) for time_step in encoded_sequence]
        time_steps = torch.stack(time_steps).reshape(-1)

        return time_steps, time_steps.shape[0]

    def get_max_seq_len(self):

        if self.max_seq_len is None:
            max_seq_len = 0
            for idx in range(len(self.sequences)):
                seq = convert_data_to_tensor(self.sequences, idx)
                if seq.numel() > 0:
                    max_seq_len = max(max_seq_len, seq.shape[0])
            self.max_seq_len = max_seq_len
        return self.max_seq_len

    def get_avg_seq_len(self):
        if self.avg_seq_len is None:
            num_pitches = 0
            for idx in range(len(self.sequences)):
                seq = convert_data_to_tensor(self.sequences, idx)
                num_pitches += seq.numel()
            self.avg_seq_len = int(num_pitches/len(self.sequences))
        return self.avg_seq_len

    def get_pitches(self):
        if len(self.pitches) == 0:

            for idx in range(len(self.sequences)):
                seq = convert_data_to_tensor(self.sequences, idx)
                self.pitches.update(seq.tolist())

        return self.pitches


    def get_max_pitch(self):
        if self.max_pitch is None:
            max_pitch = 0
            for idx in range(len(self.sequences)):
                seq = convert_data_to_tensor(self.sequences, idx)
                if seq.numel() > 0:
                    max_pitch = max(max_pitch, seq.max().item())
            self.max_pitch = max_pitch
        return self.max_pitch

    def get_min_pitch(self):
        if self.min_pitch is None:
            min_pitch = 100
            for idx in range(len(self.sequences)):
                seq = convert_data_to_tensor(self.sequences, idx)
                seq = seq[seq >= 0]
                if seq.numel() > 0:
                    min_pitch = min(min_pitch, seq.min().item())
            self.min_pitch = min_pitch
        return self.min_pitch







