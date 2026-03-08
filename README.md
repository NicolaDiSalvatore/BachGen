---
title: BachGen
emoji: 🎼
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: deploy/app.py
pinned: false
---


**BachGen** is a machine learning project for generating Bach chorales-style music.
It uses MusicTransformer architecture to generate new audio sequences and provides a FastAPI backend and Gradio UI for interaction.



## Features

- **MIDI Generation**: Generate new Bach-style chorales.
- **Audio Synthesis**: Convert generated MIDI to WAV using Fluidsynth.
- **Web Interface**: Interactive Gradio UI.
- **API**: FastAPI endpoints for serving the model.



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



## License


This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## References
Full BibTeX citations available in [`references.bib`](./references.bib).