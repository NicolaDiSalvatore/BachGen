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

- Adjustable generation length (32–2048 tokens)
- Start pitch control (MIDI note 36–84)
- Temperature slider for creativity/accuracy trade-off
- Top-k and top-p sampling parameters
- Tempo control (40–180 BPM)
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
```bash
curl -X POST "http://localhost:7860/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "n_samples": 1,
    "sequence_length": 512,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 0.9,
    "start_pitch": 60
  }'
```


## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_samples` | Number of sequences to generate | 1 |
| `sequence_length` | Length of generated sequence (tokens) | 1024 |
| `temperature` | Sampling temperature (higher = more creative) | 1.0 |
| `top_k` | Top-k sampling (0 to disable) | 0 |
| `top_p` | Top-p (nucleus) sampling probability | 0.9 |
| `start_pitch` | Starting MIDI pitch | 60 |
| `seed` | Random seed for reproducibility (0 = random) | 0 |


## Requirements

- Python 3.10+
- PyTorch 2.0+
- fluidsynth (optional, for WAV synthesis)

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [JS Bach Chorales Dataset](https://github.com/czhuang/JSB-Chorales-dataset) for training data
- [Music Transformer](https://magenta.tensorflow.org/music-transformer) for architectural inspiration
