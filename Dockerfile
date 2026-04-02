FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

ENV PYTHONPATH=/home/user/app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    fluidsynth \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache optimization)
COPY --chown=user:user requirements.txt .

# Upgrade pip (important for dependency resolution)
RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY --chown=user:user . .

# Switch user
USER user

# Expose port (Spaces uses 7860)
EXPOSE 7860

# Start app (adjust if needed)
CMD ["python", "-m", "deploy.app"]