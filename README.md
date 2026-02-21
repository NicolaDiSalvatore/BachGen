---
title: BachGen
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# BachGen

**BachGen** is a machine learning project for generating Bach chorales-style music.
It uses MusicTransformer architecture to generate new audio sequences and provides a FastAPI backend and Gradio UI for interaction.

## Features

- **MIDI Generation**: Generate new Bach-style chorales.
- **Audio Synthesis**: Convert generated MIDI to WAV using Fluidsynth.
- **Web Interface**: Interactive Gradio UI.
- **API**: FastAPI endpoints for serving the model.

## Installation

### Prerequisites

- Python 3.10+
- Fluidsynth (for audio synthesis)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/BachGen.git
   cd BachGen
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Usage

### Run the API

```bash
python api/main.py
```

### Run the UI

Open your browser at `http://localhost:7860` (assuming Gradio runs on default port).

## Docker

For deployment to Hugging Face Spaces or running in a containerized environment, see [DOCKER.md](DOCKER.md) for detailed instructions.

**Quick start:**
```bash
# Build the image
docker build -t bachgen:latest .

# Run the container
docker run -p 7860:7860 bachgen:latest

# Access at http://localhost:7860
```

## Development

### Running Tests

```bash
pytest
```

### Linting

```bash
ruff check .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
