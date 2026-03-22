
FROM python:3.14.3

WORKDIR /app

# System dependencies: audio/MIDI support + fluidsynth for synthesis
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libportmidi-dev \
    fluidsynth \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 user

# Install dependencies
COPY --chown=user:user requirements.txt .

# PyTorch CPU-only (keeps image smaller than the default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user:user . .

# Create expected runtime directories
RUN mkdir -p /app/resources /app/deploy

# Download model weights / data files
RUN python download_files.py

USER user

ENV HOME=/home/user \
    PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]