FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    fluidsynth \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user:user requirements.txt .

RUN pip install --no-cache-dir --prefer-binary torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
  pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY --chown=user:user . .

RUN mkdir -p resources deploy

RUN python download_files.py

USER user

ENV HOME=/home/user \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ROOT_PATH=/ \
    GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:7860/api/status || exit 1

CMD ["python", "app.py"]