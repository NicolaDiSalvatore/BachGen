

# Improvement ideas:
# - Add Weight Decay to Adam
# - add warmup scheduler wrapper (start small and gradually increase LR (evitate Loss spikes, unstable training, divergence)
# - label smoothing to cross entropy
# - add mixed precision training: useful when using gpu, to lower memory usage (using fp16 for most of the operations, while fp32 for critical operations ) while having the same accuracy

import argparse
import dataclasses
import json
import random
from dataclasses import asdict, dataclass, replace
from itertools import product
from os import makedirs
from os.path import join
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from mlflow.tracking import MlflowClient
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataloader import collate_fn
from src.data.dataset import BachDataset
from src.data.vocab import get_vocab_size
from src.early_stopping import EarlyStopping
from src.models.transformer import MusicTransformer

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def upload_checkpoint_to_mlflow(checkpoint_path: Path, artifact_name: str = None):
    """Upload a checkpoint file to MLflow artifacts.
    
    Args:
        checkpoint_path: Local path to the checkpoint file
        artifact_name: Optional name for the artifact subfolder (defaults to 'checkpoints')
    """
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint {checkpoint_path} does not exist, skipping upload")
        return

    artifact_folder = artifact_name or "checkpoints"
    try:
        mlflow.log_artifact(str(checkpoint_path), artifact_folder)
        print(f"Uploaded {checkpoint_path.name} to MLflow artifacts/{artifact_folder}/")
    except Exception as e:
        print(f"Warning: Failed to upload checkpoint to MLflow: {e}")


def download_checkpoint_from_mlflow(run_id: str, checkpoint_name: str, local_dir: Path) -> Path | None:
    """Download a checkpoint file from MLflow artifacts.
    
    Args:
        run_id: The MLflow run ID to download from
        checkpoint_name: Name of the checkpoint file (e.g., 'last.pth', 'best_val.pth')
        local_dir: Local directory to save the checkpoint
    
    Returns:
        Path to the downloaded checkpoint, or None if not found
    """
    client = MlflowClient()

    try:
        artifacts = client.list_artifacts(run_id, path="checkpoints")
        checkpoint_exists = any(a.path == f"checkpoints/{checkpoint_name}" for a in artifacts)

        if not checkpoint_exists:
            print(f"Checkpoint {checkpoint_name} not found in MLflow run {run_id}")
            return None


        local_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = f"checkpoints/{checkpoint_name}"

        download_path = client.download_artifacts(run_id, artifact_path, str(local_dir.parent))
        downloaded_file = Path(download_path)


        target_path = local_dir / checkpoint_name
        if downloaded_file != target_path:
            import shutil
            shutil.move(str(downloaded_file), str(target_path))

        print(f"Downloaded {checkpoint_name} from MLflow run {run_id} to {target_path}")
        return target_path

    except Exception as e:
        print(f"Warning: Failed to download checkpoint from MLflow: {e}")
        return None


def download_checkpoints_for_resume(run_id: str, checkpoints_dir: Path) -> bool:
    """Download both last.pth and best_val.pth from MLflow for resuming training.
    
    Args:
        run_id: The MLflow run ID to download from
        checkpoints_dir: Local directory to save checkpoints
    
    Returns:
        True if at least last.pth was downloaded successfully
    """
    last_path = download_checkpoint_from_mlflow(run_id, "last.pth", checkpoints_dir)
    download_checkpoint_from_mlflow(run_id, "best_val.pth", checkpoints_dir)

    return last_path is not None


@dataclass(frozen=True)
class Config:
    attention_hidden_size: int
    feedforward_hidden_size: int
    num_decoder_layers: int
    num_attention_heads: int
    epochs: int
    seed: int
    learning_rate: float
    batch_size: int
    accumulation_steps: int
    embed_dropout: float
    ffn_dropout: float
    attn_dropout: float
    attn_proj_dropout: float
    weight_decay: float


def generate_configs(config: Config, search_space: dict):
    keys = search_space.keys()
    values = search_space.values()

    for combo in product(*values):
        yield replace(config, **dict(zip(keys, combo)))


def evaluate_model(model: MusicTransformer, loss_fn: CrossEntropyLoss, loader: DataLoader, vocab_size: int):
    total_loss = 0
    total_tokens = 0

    model.eval()


    with torch.no_grad():
        for i, data in enumerate(loader):
            sequences, lengths = data
            sequences = sequences.to(DEVICE)
            inputs = sequences[:, :-1]
            targets = sequences[:, 1:]
            batch_size, seq_len = inputs.shape
            outputs = model(inputs)
            loss = loss_fn(outputs.reshape(batch_size * seq_len, vocab_size),
                           targets.reshape(batch_size * seq_len))
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss


def train_and_validate_one_epoch(
        epoch_index: int,
        writer,
        model: MusicTransformer,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        early_stopping: EarlyStopping,
        loss_fn: CrossEntropyLoss,
        checkpoints_folder: Path,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        vocab_size: int,
        accumulation_steps: int,
        scaler: GradScaler):
    total_loss = 0.0
    total_tokens = 0

    optimizer.zero_grad()
    for i, data in enumerate(training_loader):
        sequences, lengths = data
        sequences = sequences.to(DEVICE)

        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        assert torch.isfinite(inputs).all(), "NaN or Inf in dataset"
        assert inputs.dtype == torch.long, f"dtype={inputs.dtype}"
        assert inputs.min() >= 0, f"min token = {inputs.min()}"
        assert inputs.max() < vocab_size, f"max token = {inputs.max()}"

        batch_size, seq_len = inputs.shape

        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs)

            loss = loss_fn(outputs.reshape(batch_size * seq_len, vocab_size),
                            targets.reshape(batch_size * seq_len)) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(training_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


        num_tokens = targets.numel()
        total_loss += loss.item() * accumulation_steps * num_tokens
        total_tokens += num_tokens
        print(f"Loss at epoch {epoch_index} and batch {i}: {loss.item() * accumulation_steps}")

    avg_train_loss = total_loss / total_tokens
    print(f"Average training loss: {avg_train_loss}")

    writer.add_scalar("Training Loss Per Epoch", avg_train_loss, epoch_index)

    avg_validation_loss = evaluate_model(model, loss_fn, validation_loader, vocab_size)
    scheduler.step(avg_validation_loss)



    print('LOSS train {} valid {}'.format(avg_train_loss, avg_validation_loss))

    writer.add_scalar("Validation Loss Per Epoch", avg_validation_loss, epoch_index)

    torch.save({
        'epoch': epoch_index,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'avg_train_loss': avg_train_loss,
        'avg_validation_loss': avg_validation_loss,
        'attention_hidden_size': model.attention_hidden_dim,
        'feedforward_hidden_dim': model.feedforward_hidden_dim,
        'num_decoder_layers': model.num_decoder_layers,
        'num_attention_heads': model.num_attention_heads,
        'early_stopping_counter': early_stopping.counter,
        'early_stopping_best_loss': early_stopping.best_loss,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate()
        }
    }, join(checkpoints_folder, "last.pth"))

    upload_checkpoint_to_mlflow(Path(checkpoints_folder) / "last.pth")

    writer.flush()

    return model, optimizer, scheduler, avg_train_loss, avg_validation_loss


def train_and_validate(config: Config,
                       experiment_path: Path,
                       loss_fn: CrossEntropyLoss,
                       training_loader: DataLoader,
                       validation_loader: DataLoader,
                       vocab_size: int,
                       max_seq_len: int,
                       run_id: str,
                       run_path: Path = None,
                       accumulation_steps: int = 4):

    checkpoint = None
    if run_path is None:
        run_path = experiment_path / run_id
        makedirs(run_path)
    else:
        last_model_path = run_path / "checkpoints" / "last.pth"
        if last_model_path.exists():
            last_model_path = run_path / "checkpoints" / "last.pth"
            checkpoint = torch.load(last_model_path, weights_only=False, map_location=DEVICE)
        else:
            print("No checkpoint found")

    checkpoints_path = run_path / "checkpoints"
    makedirs(checkpoints_path, exist_ok=True)

    logs_path = run_path / "logs" / "tensorboard"
    makedirs(logs_path, exist_ok=True)

    with open(run_path / "config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f)

    if checkpoint is not None:
        model = MusicTransformer(seq_len=max_seq_len,
                                 vocab_size=vocab_size,
                                 attention_hidden_dim=checkpoint["attention_hidden_size"],
                                 feedforward_hidden_dim=checkpoint["feedforward_hidden_dim"],
                                 num_decoder_layers=checkpoint["num_decoder_layers"],
                                 num_attention_heads=checkpoint["num_attention_heads"],
                                 embed_dropout=config.embed_dropout,
                                 ffn_dropout=config.ffn_dropout,
                                 attn_dropout=config.attn_dropout,
                                 attn_proj_dropout=config.attn_proj_dropout
        )
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        model = MusicTransformer(seq_len=max_seq_len,
                                 vocab_size=vocab_size,
                                 attention_hidden_dim=config.attention_hidden_size,
                                 feedforward_hidden_dim=config.feedforward_hidden_size,
                                 num_decoder_layers=config.num_decoder_layers,
                                 num_attention_heads=config.num_attention_heads,
                                 embed_dropout=config.embed_dropout,
                                 ffn_dropout=config.ffn_dropout,
                                 attn_dropout=config.attn_dropout,
                                 attn_proj_dropout=config.attn_proj_dropout
        )

    model = model.to(DEVICE)
    print(f"Model moved to: {DEVICE}")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(
         optimizer,
         mode='min',
         factor=0.5,
         patience=5,
         threshold=1e-4
    )

    early_stopping = EarlyStopping(
        patience=20,
        min_delta=1e-4
    )

    writer = SummaryWriter(log_dir=str(logs_path))

    start_epoch = 1
    best_validation_loss = float("inf")
    best_epoch = None
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        early_stopping.counter = checkpoint['early_stopping_counter']
        early_stopping.best_loss = checkpoint['early_stopping_best_loss']
        print(f"Restored early stopping: counter={early_stopping.counter}, best_loss={early_stopping.best_loss}")
        best_checkpoint = torch.load(run_path / "checkpoints" / "best_val.pth", weights_only=False, map_location=DEVICE)
        best_model = MusicTransformer(seq_len=max_seq_len,
                                      vocab_size=vocab_size,
                                      attention_hidden_dim=best_checkpoint["attention_hidden_size"],
                                      feedforward_hidden_dim=best_checkpoint["feedforward_hidden_dim"],
                                      num_decoder_layers=best_checkpoint["num_decoder_layers"],
                                      num_attention_heads=best_checkpoint["num_attention_heads"],
                                      embed_dropout=config.embed_dropout,
                                      ffn_dropout=config.ffn_dropout,
                                      attn_dropout=config.attn_dropout,
                                      attn_proj_dropout=config.attn_proj_dropout
                                      )
        missing_keys, unexpected_keys = best_model.load_state_dict(best_checkpoint["model_state_dict"], strict=False)
        print(f"Best Model Missing keys: {missing_keys}")
        print(f"Best Model Unexpected keys: {unexpected_keys}")
        best_model = best_model.to(DEVICE)
        best_validation_loss = best_checkpoint["avg_validation_loss"]
        best_epoch = best_checkpoint["epoch"]

    scaler = GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(start_epoch, config.epochs+1):
        print('EPOCH {}:'.format(epoch))


        model.train(True)
        model, optimizer, scheduler, avg_train_loss, avg_validation_loss = train_and_validate_one_epoch(
            epoch, writer, model, optimizer, scheduler, early_stopping, loss_fn, checkpoints_path, training_loader, validation_loader, vocab_size, accumulation_steps, scaler)

        early_stopping(avg_validation_loss)

        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            best_model = model
            best_epoch = epoch

            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'avg_train_loss': avg_train_loss,
                'avg_validation_loss': avg_validation_loss,
                'attention_hidden_size': config.attention_hidden_size,
                'feedforward_hidden_dim': config.feedforward_hidden_size,
                'num_decoder_layers': config.num_decoder_layers,
                'num_attention_heads': config.num_attention_heads,
                "rng_state": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate()
                }
            }, join(checkpoints_path, "best_val.pth"))


            upload_checkpoint_to_mlflow(Path(checkpoints_path) / "best_val.pth")

        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_validation_loss, step=epoch)


    writer.close()

    return best_model, best_validation_loss, best_epoch


def get_latest_run_path(experiment_path: Path) -> Path:
    runs = sorted(p for p in experiment_path.iterdir())
    return Path(experiment_path / runs[-1])


def start_or_resume_run(run_id: str | None = None):
    if mlflow.active_run() is not None:
        mlflow.end_run()

    if run_id is not None:
        client = MlflowClient()
        client.get_run(run_id)
        return mlflow.start_run(run_id=run_id)

    else:
        return mlflow.start_run()


def get_starting_config_id(run_id: str | None = None, starting_config: int = None):
    if run_id is not None:
        run = mlflow.get_run(run_id)
        config_id = run.data.tags.get("config_id")
        return int(config_id) if config_id is not None else 0

    if starting_config is not None:
       return starting_config

    return 0


def prepare_run_path(experiment_path: Path, run_id: str) -> Path:
    run_path = experiment_path / run_id
    (run_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str,
                        help="MLflow run ID to resume from. If provided, downloads checkpoints from MLflow.")
    parser.add_argument("--starting_config", type=int)
    parser.add_argument("--mlflow-uri", type=str, default="sqlite:///mlflow.db",
                        help="MLflow tracking URI (default: sqlite:///mlflow.db)")
    parser.add_argument("--config", type=str, default="src/config/search_space.yaml",
                        help="Path to the YAML configuration file.")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_training_set = BachDataset(split='train')
    min_pitch = raw_training_set.get_min_pitch()
    max_pitch = raw_training_set.get_max_pitch()


    AUGMENT_RANGE = 6
    min_pitch_aug = min_pitch - AUGMENT_RANGE
    max_pitch_aug = max_pitch + AUGMENT_RANGE

    training_set = BachDataset(split='train', min_pitch=min_pitch_aug, augment=True)
    validation_set = BachDataset(split='valid', min_pitch=min_pitch_aug, augment=False)
    test_set = BachDataset(split='test', min_pitch=min_pitch_aug, augment=False)

    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
        search_space = config_data["search_space"]

    base_config = Config(num_attention_heads=search_space["num_attention_heads"][0],
                         num_decoder_layers=search_space["num_decoder_layers"][0],
                         attention_hidden_size=search_space["attention_hidden_size"][0],
                         feedforward_hidden_size=search_space["feedforward_hidden_size"][0],
                         epochs=search_space["epochs"][0],
                         seed=search_space["seed"][0],
                         learning_rate=search_space["learning_rate"][0],
                         batch_size=search_space["batch_size"][0],
                         accumulation_steps=search_space["accumulation_steps"][0],
                         embed_dropout=search_space["embed_dropout"][0],
                         ffn_dropout=search_space["ffn_dropout"][0],
                         attn_dropout=search_space["attn_dropout"][0],
                         attn_proj_dropout=search_space["attn_proj_dropout"][0],
                         weight_decay=search_space["weight_decay"][0])

    configs = list(generate_configs(base_config, search_space))
    print(f"Number of configs: {len(configs)}")

    project_path = Path(__file__).resolve().parents[1]
    print(f"project_path: {project_path}")

    mlflow.set_tracking_uri(args.mlflow_uri)

    experiment_name = "transformer_bach_dataset"
    mlflow.set_experiment(experiment_name)

    experiment_path = project_path / "experiments" / experiment_name
    makedirs(experiment_path, exist_ok=True)

    run_path, run_id = None, None

    if args.run is not None:
        run_id = args.run

        client = MlflowClient()
        try:
            client.get_run(run_id)
            print(f"Found MLflow run: {run_id}")
        except Exception as e:
            raise ValueError(f"MLflow run {run_id} not found: {e}")

        run_path = experiment_path / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = run_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading checkpoints from MLflow run {run_id}...")
        if not download_checkpoints_for_resume(run_id, checkpoints_dir):
            raise FileNotFoundError(
                f"Could not download checkpoints from MLflow run {run_id}. "
                f"Make sure the run has 'checkpoints/last.pth' artifact."
            )

        print(f"Resuming from MLflow run: {run_id}")

    starting_config = None

    if args.starting_config and args.run is not None:
        raise ValueError(
            "multiple initial starting configuration"
        )

    if args.starting_config:
        starting_config = args.starting_config


    for i, config in enumerate(configs, start=get_starting_config_id(run_id, starting_config)):

        num_workers = 4 if torch.cuda.is_available() else 0
        pin_memory = True if torch.cuda.is_available() else False

        training_loader = DataLoader(training_set, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
        validation_loader = DataLoader(validation_set, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
        max_seq_len = max(training_set.get_max_seq_len(), validation_set.get_max_seq_len(), test_set.get_max_seq_len())
        vocab_size = get_vocab_size(min_pitch=min_pitch_aug, max_pitch=max_pitch_aug)

        print('Training set has {} instances'.format(len(training_set)))
        # for i, data in enumerate(training_loader):
        #     seq, lengths = data
        #     print(f"Iter {i} lengths: {lengths}")
        #     print(f"Iter {i} shape: {seq.shape}")
        print('Validation set has {} instances'.format(len(validation_set)))
        print('Test set has {} instances'.format(len(test_set)))
        print(f"Max seq length: {max_seq_len}")
        print(f"Max pitch: {max_pitch}")
        print(f"Min pitch: {min_pitch}")
        print(f"Augmented Max pitch: {max_pitch_aug}")
        print(f"Augmented Min pitch: {min_pitch_aug}")
        print(f"Pad token: {0}")
        print(f"Rest token: {1}")
        print(f"Vocab size: {vocab_size}")
        print(f"CONFIG {i}")
        best_overall_validation_loss = float("inf")
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        with start_or_resume_run(run_id) as run:
            if args.run is not None and get_starting_config_id(run_id) == i:
                print(f"Resuming from config {i} and run_id {run.info.run_id}")
            else:
                print(f"Starting new run with id {run.info.run_id}")
            loss_fn = CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
            params = asdict(config)
            try:

                sanitized_params = {k: str(v) for k, v in params.items()}
                mlflow.log_params(sanitized_params)
            except Exception as e:
                if "INVALID_PARAMETER_VALUE" in str(e):
                     print("Warning: Parameters already logged for this run. Skipping log_params.")
                else:
                    print(f"Failed to log params. Params: {params}")
                    print(f"Error: {e}")
                    raise e

            mlflow.set_tag("config_id", str(i))

            best_model, best_validation_loss, best_epoch = train_and_validate(
                config, experiment_path, loss_fn, training_loader, validation_loader, vocab_size, max_seq_len, run.info.run_id, run_path, config.accumulation_steps
            )

            mlflow.pytorch.log_model(best_model, name=f"best_model_config{i}")
            mlflow.log_metric("best_validation_loss", best_validation_loss)
            mlflow.set_tag("epoch_of_best_validation_loss", str(best_epoch))
            mlflow.set_tag("run_id", run.info.run_id)
            mlflow.set_tag("config_id", str(i))
            mlflow.set_tag("saved_model", "best_val")
            mlflow.set_tag("selection_metric", "val_loss")

            if best_validation_loss < best_overall_validation_loss:
                best_overall_config = i
                best_overall_epoch = best_epoch
                best_run_id = run.info.run_id
                best_overall_validation_loss = best_validation_loss
                best_overall_model = best_model
                mlflow.pytorch.log_model(best_model, name="best_model")
                mlflow.log_metric("best_validation_loss", best_validation_loss)
                mlflow.set_tag(f"run_id_config{i}", run.info.run_id)

            run_id = None

    print(f"run_id of the best model: {best_run_id}")

    experiment_name = "transformer_bach_dataset"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        avg_test_loss = evaluate_model(best_overall_model, loss_fn, test_loader, vocab_size)
        mlflow.pytorch.log_model(
            best_overall_model,
            artifact_path="best_model",
            registered_model_name="TransformerBachDataset"
        )

        mlflow.set_tag("epoch_of_best_overall_validation_loss", str(best_overall_epoch))
        mlflow.set_tag("run_id_of_best_overall_model", run.info.run_id)
        mlflow.set_tag("config_id", str(best_overall_config))
        mlflow.log_metric("validation_loss", best_overall_validation_loss)
        mlflow.log_metrics({
            "test_loss": avg_test_loss
        })


if __name__ == "__main__":
    main()
