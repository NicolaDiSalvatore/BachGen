---
title: BachGen
emoji: 🎵
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Generate Bach chorale-style music with Transformer models
---

BachGen is a music generation system that creates polyphonic chorales in the style of J.S. Bach. Built on a decoder-only Music Transformer architecture-voice harmonic progressions, it generates four token by token, producing MIDI files that capture the essence of Bach's compositional style.

## Features

- **Music Transformer Architecture** — Decoder-only Transformer trained on the JS Bach chorale dataset
- **Flexible Sampling** — Temperature, top-k, and top-p (nucleus) sampling for controllable generation
- **MIDI Output** — Generate downloadable MIDI files with configurable tempo
- **Audio Synthesis** — Optional WAV rendering via fluidsynth for immediate playback
- **REST API** — FastAPI-powered backend for programmatic generation
- **Web Interface** — Gradio-powered UI for easy interaction

## Hugging Face Space

[**Live Demo →**](https://huggingface.co/spaces/NicolaDiSalvatore/BachGen)

The easiest way to use BachGen is through our Hugging Face Space, which provides a built-in Gradio interface:

- Adjustable generation length
- Start pitch control
- Temperature slider for creativity/accuracy trade-off
- Top-k and top-p sampling parameters
- Tempo control
- MIDI download and optional audio playback

## Quick Start

### Local Deployment

```bash
pip install -r requirements.txt
python download_files.py
python -m uvicorn deploy.app:app --host 0.0.0.0 --port 7860
```

### Docker Deployment

```bash
docker build -t bachgen .
docker run -p 7860:7860 bachgen
```

### API Usage

**Linux/Mac:**
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

iwr -Uri "http://localhost:7860/api/download/$run_id" -OutFile "bachgen_output.zip"
```



## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_samples` | Number of sequences to generate | 1       |
| `sequence_length` | Length of generated sequence (tokens) | 256     |
| `temperature` | Sampling temperature (higher = more creative) | 1.0     |
| `top_k` | Top-k sampling (0 to disable) | 0       |
| `top_p` | Top-p (nucleus) sampling probability | 0.9     |
| `start_pitch` | Starting MIDI pitch | 60      |
| `seed` | Random seed for reproducibility (0 = random) | 0       |


## Requirements

- Python 3.10+
- PyTorch 2.0+
- fluidsynth

## License

MIT License - see LICENSE file for details.

## References
Full BibTeX citations available in [`references.bib`](./references.bib).
### Papers
- Huang, C.-Z. A., Vaswani, A., Uszkoreit, J., Shazeer, N., Simon, I., Hawthorne, C., Dai, A. M.,
  Hoffman, M. D., Dinculescu, M., & Eck, D. (2018). *Music Transformer*. arXiv:1809.04281.
  https://arxiv.org/abs/1809.04281

### Dataset
- Boulanger-Lewandowski, N., Bengio, Y., & Vincent, P. (2012). *Modeling Temporal Dependencies
  in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription*.
  arXiv:1206.6392. https://arxiv.org/abs/1206.6392

### Soundfont
- *Florestan Ahh Choir* [SF2 soundfont]. Public domain.
  https://musical-artifacts.com/artifacts/388