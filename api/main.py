import logging
import os
import subprocess
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from os.path import abspath, dirname
from pathlib import Path

import gradio as gr
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rendering.midi import sequences_to_midi
from src.generate import generate_sequences
from src.models.transformer import MusicTransformer

load_dotenv()


def synthesize_midi(midi_path: Path, output_wav_path: Path):
    """
    Synthesize MIDI to WAV using fluidsynth
    """
    soundfont_path = Path(os.getenv("SOUNDFONT_PATH", str(project_path / "resources" / "052_Florestan_Ahh_Choir.sf2")))

    if not soundfont_path.exists():
        logger.warning(f"SoundFont not found at {soundfont_path}")
        logger.warning("Set SOUNDFONT_PATH environment variable")
        return False

    try:
        cmd = [
            "fluidsynth",
            "-F",
            str(output_wav_path),
            str(soundfont_path),
            str(midi_path),
        ]

        logger.info(f"Running synthesis: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Fluidsynth failed: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except FileNotFoundError:
        logger.warning("Fluidsynth executable not found. Skipping synthesis.")
        return False
    except Exception as e:
        logger.exception(f"Synthesis failed: {e}")
        return False


app = FastAPI(
    title="BachGen API",
    description="Generate Bach-style chorales as MIDI files",
    version="1.0.0",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transformer_bach_dataset")

project_path = Path(dirname(dirname(abspath(__file__))))
logging.info(f"project_path: {project_path}")

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    local_model_path = Path(project_path) / "deploy" / "model.pth"
    if not local_model_path.exists():
        raise ValueError(f"Model in path {local_model_path} not found")

    if local_model_path.exists():
        logger.info(f"Loading local model from {local_model_path}...")
        try:
            checkpoint = torch.load(
                local_model_path, map_location=torch.device("cpu"), weights_only=False
            )

            if (
                isinstance(checkpoint, dict)
                and "config" in checkpoint
                and "model_state_dict" in checkpoint
            ):
                config = checkpoint["config"]
                logger.info(f"Loading model with config: {config}")

                model = MusicTransformer(
                    vocab_size=config.get("vocab_size", 92),
                    seq_len=config.get("seq_len", 2048),
                    attention_hidden_dim=config.get(
                        "attention_hidden_size", 512
                    ),
                    feedforward_hidden_dim=config.get("feedforward_hidden_dim", 2048),
                    num_decoder_layers=config.get("num_decoder_layers", 6),
                    num_attention_heads=config.get("num_attention_heads", 8),
                    embed_dropout=config.get("embed_dropout", 0.0),
                    ffn_dropout=config.get("ffn_dropout", 0.0),
                    attn_dropout=config.get("attn_dropout", 0.0),
                    attn_proj_dropout=config.get("attn_proj_dropout", 0.0),
                )

                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            else:
                logger.warning(
                    "Unknown checkpoint format. Attempting to load as state dict with default config (RISKY)."
                )

                raise ValueError("Invalid checkpoint format")

            model.eval()
            logger.info("Local model loaded successfully!")

        except Exception as e:
            logger.exception(f"Failed to load local model: {e}")
            raise

    else:
        logger.error(
            f"Model file not found at {local_model_path}. Please ensure model.pth is present."
        )
        raise FileNotFoundError(f"Model file not found at {local_model_path}")

    yield

    logger.info("Cleaning up...")
    model = None


app = FastAPI(lifespan=lifespan)


MAX_SEQUENCE_LENGTH = 2048
MAX_SAMPLES = 16


class GenerateRequest(BaseModel):
    n_samples: int = Field(1, ge=1, le=MAX_SAMPLES)
    sequence_length: int = Field(1024, ge=32, le=MAX_SEQUENCE_LENGTH)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    top_k: int = Field(0, ge=0, description="Top-k sampling (0 to disable)")
    top_p: float = Field(
        0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling (0.0 to disable)"
    )
    start_pitch: int = Field(60, ge=21, le=108, description="MIDI start pitch")
    seed: int = Field(None, ge=0, description="Random seed for reproducibility (optional)")


class GenerateResponse(BaseModel):
    run_id: str
    n_samples: int
    sequence_length: int
    temperature: float
    top_k: int
    top_p: float
    start_pitch: int
    seed: int = None
    download_url: str


logger = logging.getLogger("transformer_bach_dataset")
logging.basicConfig(level=logging.INFO)


@app.get("/api/status")
def root():
    return {"status": "BachGen API running"}


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    logger.info(f"Starting generation run {run_id}")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"transformer_bach_dataset_{run_id}"))

    try:
        sequences = generate_sequences(
            model=model,
            length=req.sequence_length,
            start_midi_pitch=req.start_pitch,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            num_sequences=req.n_samples,
            seed=req.seed,
        )

        midi_paths = sequences_to_midi(sequences, tmp_dir, return_output_paths=True)

        zip_path = tmp_dir / "generated_midis.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            for p in midi_paths:
                z.write(p, arcname=p.name)

        logger.info(f"Generation completed for run {run_id}")

        return GenerateResponse(
            run_id=run_id,
            n_samples=req.n_samples,
            sequence_length=req.sequence_length,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            start_pitch=req.start_pitch,
            seed=req.seed,
            download_url=f"/download/{run_id}",
        )

    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{run_id}")
def download(run_id: str):

    tmp_root = Path(tempfile.gettempdir())
    matches = list(tmp_root.glob(f"transformer_bach_dataset_{run_id}*"))

    if not matches:
        raise HTTPException(status_code=404, detail="Run not found")

    zip_path = matches[0] / "generated_midis.zip"

    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="ZIP file missing")

    return FileResponse(
        zip_path, media_type="application/zip", filename="generated_midis.zip"
    )


def gradio_generate(length_tokens, temperature, top_k, top_p, start_pitch, tempo_bpm, seed):
    """
    Wrapper for Gradio interface
    """
    global model
    if model is None:
        raise gr.Error("Model not loaded yet. Please wait a moment.")

    try:
        tempo_value = int(tempo_bpm)
        logger.info(f"Generating with tempo_bpm: {tempo_value}, seed: {seed}")
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"gradio_bach_{run_id}"))

        seed_value = int(seed) if seed is not None and seed > 0 else None
        
        sequences = generate_sequences(
            model=model,
            length=int(length_tokens),
            start_midi_pitch=int(start_pitch),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            num_sequences=1,
            seed=seed_value,
        )

        midi_paths = sequences_to_midi(sequences, tmp_dir, return_output_paths=True, tempo_bpm=tempo_value)

        if not midi_paths:
            raise gr.Error("No sequences generated")

        midi_path = midi_paths[0]
        wav_path = tmp_dir / f"{midi_path.stem}.wav"

        if synthesize_midi(midi_path, wav_path):
            return str(midi_path), str(wav_path)
        else:
            return str(midi_path), None

    except Exception as e:
        logger.exception("Gradio generation failed")
        raise gr.Error(f"Generation failed: {str(e)}")


with gr.Blocks(title="Bach Chorale Generator") as demo:
    gr.Markdown("# Bach Chorale Generator (Transformer)")
    gr.Markdown("Generate infinite polyphonic music in the style of J.S. Bach.")

    with gr.Row():
        with gr.Column():
            length_input = gr.Number(
                label="Length (Tokens)", value=256, minimum=32, step=16
            )
            start_pitch_slider = gr.Slider(
                minimum=36, maximum=84, value=60, step=1, label="Start Pitch (MIDI)"
            )
            temp_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature (Creativity)",
            )
            top_k_slider = gr.Slider(
                minimum=0, maximum=100, value=0, step=1, label="Top-K (0 to disable)"
            )
            top_p_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.9, step=0.01, label="Top-P (Nucleus)"
            )
            tempo_slider = gr.Slider(
                minimum=40, maximum=180, value=50, step=1, label="Tempo (BPM)"
            )
            seed_input = gr.Number(
                label="Seed (0 for random)", value=0, minimum=0, step=1
            )
            gen_btn = gr.Button("Generate Music", variant="primary")

        with gr.Column():
            midi_out = gr.File(label="Download MIDI")
            audio_out = gr.Audio(label="Play Generated MIDI", type="filepath")

    gen_btn.click(
        fn=gradio_generate,
        inputs=[
            length_input,
            temp_slider,
            top_k_slider,
            top_p_slider,
            start_pitch_slider,
            tempo_slider,
            seed_input,
        ],
        outputs=[midi_out, audio_out],
    )

app = gr.mount_gradio_app(app, demo, path="/")
