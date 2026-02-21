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
   git clone https://github.com/NicolaDiSalvatore/BachGen
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
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Run the UI

Open your browser at `http://localhost:8000/gradio`.

## Docker


**Quick start:**
```bash
docker build -t bachgen:latest .

docker run -p 7860:7860 bachgen:latest

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
