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


    artifact_path = "checkpoints/best_val.pth"

    try:

        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, artifact_path)
        print(f"Downloaded artifact to: {local_path}")


        os.makedirs(os.path.dirname(output_path), exist_ok=True)


        checkpoint = torch.load(local_path, map_location=torch.device('cpu'))


        export_data = {
            'model_state_dict': checkpoint['model_state_dict'],
            'config': {
                'attention_hidden_size': checkpoint['attention_hidden_size'],
                'feedforward_hidden_dim': checkpoint['feedforward_hidden_dim'],
                'num_decoder_layers': checkpoint['num_decoder_layers'],
                'num_attention_heads': checkpoint['num_attention_heads'],
                'embed_dropout': 0.0,
                'ffn_dropout': 0.0,
                'attn_dropout': 0.0,
                'attn_proj_dropout': 0.0,
                'vocab_size': 92,
                'seq_len': 2048
            }
        }

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
