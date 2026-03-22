RUN useradd -m -u 1000 user

COPY --chown=user:user requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user ../Downloads/cf .

RUN mkdir -p /app/resources /app/deploy \
    && chown -R user:user /app

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