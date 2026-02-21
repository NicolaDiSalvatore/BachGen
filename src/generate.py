import argparse
import json
import os
import random
from datetime import datetime
from os.path import abspath, dirname
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.tracking import MlflowClient

from rendering.midi import sequences_to_midi
from src.data.dataset import BachDataset
from src.data.vocab import encode_pitch
from src.models.transformer import MusicTransformer

# improvement: start pitches try to put 4 values (SATB) and the average present in the training set

def generate_sequences(model: MusicTransformer, length: int, start_midi_pitch=60, temperature=1.0, top_k=0, top_p=0.9, num_sequences: int = 1):
    """
    Generates a music sequence using the model.
    start_sequence: tensor or list of starting tokens
    """
    min_pitch = BachDataset(split='train').get_min_pitch() - 6
    start_token = encode_pitch(start_midi_pitch, min_pitch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sequences = []
    for i in range(num_sequences):
        model.eval()
        sequence = [start_token]

        with torch.no_grad():
            for _ in range(length-1):
                input_tensor = torch.tensor([sequence], device=device)
                logits = model(input_tensor)[:, -1, :]
                logits = logits / temperature


                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, top_k)
                    min_val = top_k_vals[:, -1]
                    logits[logits < min_val.unsqueeze(1)] = float('-inf')

                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p

                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                sequence.append(next_token)

        rest_token = 1
        pad_token = 0
        decoded_sequence = []
        for p in sequence:
            if p == rest_token or p == pad_token:
                decoded_sequence.append(-1)
            else:
                # encode_pitch: pitch - min_pitch + 2
                # decode: token - 2 + min_pitch
                decoded_sequence.append(int(p - 2 + min_pitch))

        sequence = torch.tensor(decoded_sequence)
        # print(f"Sequence before reshape: {sequence}")
        # Truncate to multiple of 4 if needed
        if len(sequence) % 4 != 0:
            sequence = sequence[:-(len(sequence) % 4)]
        sequence = sequence.reshape(-1, 4)
        # print(f"Sequence after reshape: {sequence}")
        sequences.append(sequence)
    return sequences


def save_config(config: dict, output_path: Path, timestamp: str) -> None:
    config_path = Path(output_path / "configs" / f"{timestamp}.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    print(f"Sample saved to {config_path}")


def save_sequences(sequences: list, output_path: Path, timestamp: str) -> None:
    sequences_path = Path(output_path / "sequences")
    for i, seq in enumerate(sequences):
        print(seq.dtype)
        torch.save(seq, os.path.join(sequences_path, f"{timestamp}_sample{i + 1}.pt"))

    print(f"Sequences saved to {sequences_path}")


def generate_music(num_sequences: int, sequence_length: int, temperature: float, top_k: int, top_p: float, seed: int, start_midi_pitch: int = 60, return_sequences: bool = False):
    project_path = Path(dirname(dirname(abspath(__file__))))
    print(f"project_path: {project_path}")


    print(f"Sequence length: {sequence_length}")
    # mlflow.set_tracking_uri("file:///C:/Users/nicol/OneDrive/Projects/BachGen_raw/mlruns")
    experiment_name = "transformer_bach_dataset"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' does not exist.")
    mlflow.set_experiment(experiment_name)

    # mlflow.set_tracking_uri(f"file:///{project_path / 'mlruns'}")
    mlflow.set_registry_uri(f"file:///{project_path / 'mlflow_db' / 'mlflow.db'}")

    # best_run_id = "89fdf40d1c9e46439c4098de7aea02d7"
    # model_uri = f"runs:/{best_run_id}/best_model"

    # model = mlflow.pytorch.load_model(
    #    model_uri
    # )

    client = MlflowClient()
    for m in client.search_registered_models():
        print("Registered model:", m.name)
        versions = client.search_model_versions(f"name='{m.name}'")
        for v in versions:
            print(f"Version: {v.version}, Stage: {v.current_stage}, Run ID: {v.run_id}")

    model_uri = "models:/TransformerBachDatasetModel/3"
    model = mlflow.pytorch.load_model(model_uri)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = Path(project_path / "outputs")

    sequences = generate_sequences(model, length=sequence_length, start_midi_pitch=start_midi_pitch, temperature=temperature, top_k=top_k, top_p=top_p, num_sequences=num_sequences)

    sequences_to_midi(sequences, outputs_dir, timestamp)

    config = {'num_sequences': num_sequences,
              'sequence_length': sequence_length,
              'temperature': temperature,
              'top_k': top_k,
              'top_p': top_p,
              'seed': seed,
              'start_midi_pitch': start_midi_pitch,
              'model_uri': model_uri}

    save_config(config, outputs_dir, timestamp)
    save_sequences(sequences, outputs_dir, timestamp)


def main():

    raw_training_set = BachDataset(split='train')

    parser = argparse.ArgumentParser(description="Generate music with trained MusicTransformer")
    # parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID of best model")
    parser.add_argument("--num_sequences", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=raw_training_set.get_avg_seq_len())
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling (0.0 to disable)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_pitch", type=int, default=60, help="MIDI pitch to start generation (default: 60 Middle C)")
    args = parser.parse_args()

    generate_music(args.num_sequences, args.sequence_length, args.temperature, args.top_k, args.top_p, args.seed, start_midi_pitch=args.start_pitch)


if __name__ == "__main__":
    main()
