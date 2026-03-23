# BachGen

[![BachGen CI](https://github.com/NicolaDiSalvatore/BachGen/actions/workflows/ci.yml/badge.svg)](https://github.com/NicolaDiSalvatore/BachGen/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.14.3-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-live%20demo-yellow)](https://huggingface.co/spaces/NicolaDiSalvatore/BachGen)

**BachGen** generates Bach chorale-style music using a custom MusicTransformer. It encodes 4-voice SATB chorales as pitch token sequences and trains with next-token prediction, producing MIDI output you can download or play back directly.

> **Try it now →** [huggingface.co/spaces/NicolaDiSalvatore/BachGen](https://huggingface.co/spaces/NicolaDiSalvatore/BachGen)

---

## Architecture

BachGen uses a **MusicTransformer** built from scratch:

```
SATB chorale → pitch tokenisation → sinusoidal positional encoding
    → causal transformer (cross-entropy loss) → autoregressive sampling → MIDI
```

- 4-voice SATB input encoded as flat pitch token sequences
- Causal masking for autoregressive generation
- Pitch augmentation during training for transposition robustness
- Nucleus (top-p) and top-k sampling at inference

---

## Prerequisites

- Python 3.11+
- [Fluidsynth](https://www.fluidsynth.org/) (for WAV audio synthesis — optional, MIDI always works)
- A SoundFont file (set `SOUNDFONT_PATH` env variable, or use the default path)

---

## Quick Start

### Hugging Face Space (no install)

The easiest way — go to the [live demo](https://huggingface.co/spaces/NicolaDiSalvatore/BachGen) and generate directly in your browser.

### Local deployment

```bash
pip install -r requirements.txt
python download_files.py        # downloads model weights
python -m uvicorn deploy.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t bachgen .
docker run -p 7860:7860 bachgen
```

### Training

```bash
# fresh run
python -m src.train --config src/config/search_space.yaml --mlflow-uri sqlite:///mlflow.db

# resume from checkpoint
python -m src.train --run <mlflow_run_id>
```

---

## API

### Generate

**Linux / macOS:**
```bash
run_id=$(curl -s -X POST "http://localhost:7860/api/generate" \
  -H "Content-Type: application/json" \
  -d '{"n_samples":1,"sequence_length":256,"temperature":1.0,"top_k":0,"top_p":0.9,"start_pitch":60,"seed":0}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")

curl -O "http://localhost:7860/api/download/$run_id"
```

**Windows (PowerShell):**
```powershell
$response = Invoke-WebRequest -Uri "http://localhost:7860/api/generate" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"n_samples":1,"sequence_length":256,"temperature":1.0,"top_k":0,"top_p":0.9,"start_pitch":60,"seed":0}'

$run_id = ($response.Content | ConvertFrom-Json).run_id
Invoke-WebRequest -Uri "http://localhost:7860/api/download/$run_id" -OutFile "bachgen_output.zip"
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_samples` | Number of sequences to generate | `1` |
| `sequence_length` | Length of generated sequence (tokens) | `256` |
| `temperature` | Sampling temperature — higher = more creative | `1.0` |
| `top_k` | Top-k sampling (`0` to disable) | `0` |
| `top_p` | Nucleus sampling probability | `0.9` |
| `start_pitch` | Starting MIDI pitch (60 = middle C) | `60` |
| `seed` | Random seed (`0` = random) | `0` |

---

## Project Structure

```
src/
├── train.py              # Training script with MLflow tracking
├── generate.py           # Inference and sampling
├── data/                 # Dataset, dataloader, vocab
├── models/               # MusicTransformer architecture
├── layers/               # Custom layer implementations
api/                      # FastAPI backend
deploy/                   # FastAPI app entry point
tests/                    # Unit tests
```

---

## Testing

```bash
pytest tests/ -v
```

---

## References

- Huang et al. (2018) — [Music Transformer](https://arxiv.org/abs/1809.04281)
- Bach chorales dataset — [JSB Chorales](http://www-etud.iro.umontreal.ca/~boulanni/icml2012)

Full BibTeX citations in [`references.bib`](./references.bib).

---

## License

MIT — see [LICENSE](LICENSE).
