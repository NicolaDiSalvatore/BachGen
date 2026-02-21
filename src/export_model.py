import argparse
import os

import mlflow
import torch


def export_best_model(run_id=None, output_path="deploy/model.pth"):
    """
    Exports the best model from MLflow to a standalone file.
    If run_id is None, it tries to find the best run from the experiment.
    """

    mlflow.set_tracking_uri("sqlite:///mlflow_db/mlflow.db")
    experiment_name = "transformer_bach_dataset"
    mlflow.set_experiment(experiment_name)

    if run_id is None:
        print(f"Searching for best model in experiment '{experiment_name}'...")
        # Search for runs, ordering by validation loss
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["metrics.val_loss ASC"],
            max_results=1
        )

        if runs.empty:
            print("No runs found in experiment.")
            return False

        best_run = runs.iloc[0]
        run_id = best_run.run_id
        val_loss = best_run["metrics.val_loss"]
        print(f"Found best run: {run_id} with validation loss: {val_loss}")
    else:
        print(f"Using specified run_id: {run_id}")

    # Artifact path in MLflow (usually 'checkpoints/best_val.pth' based on your train.py)
    artifact_path = "checkpoints/best_val.pth"

    try:
        # Download the artifact
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, artifact_path)
        print(f"Downloaded artifact to: {local_path}")

        # Ensure deploy directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load the checkpoint to get the model state_dict and config
        checkpoint = torch.load(local_path, map_location=torch.device('cpu'))

        # We need to save the model structure config AND the weights
        # because we're going to load it without the original training code dependency if possible,
        # or at least in a clean way.
        # Ideally, we save the full model or just the state_dict + config.
        # Let's save a consolidated dictionary that has everything needed for inference.

        export_data = {
            'model_state_dict': checkpoint['model_state_dict'],
            'config': {
                'attention_hidden_size': checkpoint['attention_hidden_size'],
                'feedforward_hidden_dim': checkpoint['feedforward_hidden_dim'], # Note: naming mismatch in train.py (dim vs size)
                'num_decoder_layers': checkpoint['num_decoder_layers'],
                'num_attention_heads': checkpoint['num_attention_heads'],
                # Add default dropout values needed for model init, though not used in inference
                'embed_dropout': 0.0,
                'ffn_dropout': 0.0,
                'attn_dropout': 0.0,
                'attn_proj_dropout': 0.0,
                'vocab_size': 92, # Standard Bach chorales vocab size roughly, but better to fetch from somewhere if variable
                'seq_len': 2048   # Inference max length
            }
        }

        # Save to deploy folder
        torch.save(export_data, output_path)
        print(f"Successfully exported model to {output_path}")
        return True

    except Exception as e:
        print(f"Error exporting model: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, help="Specific MLflow run ID to export")
    args = parser.parse_args()

    export_best_model(args.run_id)
