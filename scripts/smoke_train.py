import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda import is_available

from src.data.dataset import BachDataset
from src.data.dataloader import collate_fn
from src.data.vocab import get_vocab_size
from src.models.transformer import MusicTransformer

DEVICE = torch.device("cuda" if is_available() else "cpu")

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "smoke_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw_training_set = BachDataset(split="train")
min_pitch = raw_training_set.get_min_pitch()
max_pitch = raw_training_set.get_max_pitch()
AUGMENT_RANGE = 6
min_pitch_aug = min_pitch - AUGMENT_RANGE
max_pitch_aug = max_pitch + AUGMENT_RANGE

training_set = BachDataset(split="train", min_pitch=min_pitch_aug, augment=True)
smoke_subset = torch.utils.data.Subset(training_set, range(min(10, len(training_set))))
smoke_loader = DataLoader(smoke_subset, batch_size=2, collate_fn=collate_fn, num_workers=0)

max_seq_len = max(smoke_subset[i][1] * 4 for i in range(len(smoke_subset)))
vocab_size = get_vocab_size(min_pitch=min_pitch_aug, max_pitch=max_pitch_aug)

model = MusicTransformer(
    seq_len=max_seq_len,
    vocab_size=vocab_size,
    attention_hidden_dim=128,
    feedforward_hidden_dim=256,
    num_decoder_layers=2,
    num_attention_heads=4,
    embed_dropout=0.0,
    ffn_dropout=0.0,
    attn_dropout=0.0,
    attn_proj_dropout=0.0,
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn = CrossEntropyLoss(ignore_index=0)

model.train()
for batch_idx, (sequences, lengths) in enumerate(smoke_loader):
    if batch_idx >= 1:
        break
    sequences = sequences.to(DEVICE)
    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]
    outputs = model(inputs)
    loss = loss_fn(
        outputs.reshape(outputs.size(0) * outputs.size(1), vocab_size),
        targets.reshape(targets.size(0) * targets.size(1))
    )
    assert torch.isfinite(loss), "Loss is NaN or Inf"
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Smoke train batch loss: {loss.item():.4f}")

torch.save(model.state_dict(), OUTPUT_DIR / "smoke_model.pth")
print(f"Smoke model saved to {OUTPUT_DIR / 'smoke_model.pth'}")
