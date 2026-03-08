FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libportmidi-dev \
    fluidsynth \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

COPY --chown=user:user requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

RUN mkdir -p /app/resources /app/deploy

RUN python download_files.py

USER user
ENV HOME=/home/user \
    PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
