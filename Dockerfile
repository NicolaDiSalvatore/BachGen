FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libportmidi-dev \
    fluidsynth \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user

# Copy requirements.txt
COPY --chown=user:user requirements.txt .

# Install PyTorch first (CPU version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY --chown=user:user . .

# Create directories for resources and deploy
RUN mkdir -p /app/resources /app/deploy

# Download any needed files
RUN python download_files.py

# Set user and environment variables
USER user
ENV HOME=/home/user \
    PYTHONPATH=/app

# Expose the port
EXPOSE 7860

# Start the FastAPI app
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]