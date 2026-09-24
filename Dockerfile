# AegisAI Command Center - hosted (cloud-mode) image.
# No camera on a server, so Live Vision and its torch/ultralytics stack are
# left out (requirements-cloud.txt). One gunicorn worker is required: all live
# state is in process memory (see "Production server" in ROADMAP.md).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AEGISAI_CLOUD_MODE=true \
    AEGISAI_TRUST_PROXY=true

WORKDIR /app

# Dependencies first so code edits don't invalidate this layer.
COPY command_center/requirements-cloud.txt command_center/requirements-cloud.txt
RUN pip install --no-cache-dir -r command_center/requirements-cloud.txt

COPY pyproject.toml ./
COPY aegis_core ./aegis_core
RUN pip install --no-cache-dir .

COPY command_center ./command_center

RUN useradd --create-home --uid 10001 aegis
USER aegis

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz', timeout=4)"

# Cloud mode refuses to start without AEGISAI_SECRET_KEY and AEGISAI_PASSWORD_HASH.
CMD ["sh", "-c", "exec gunicorn -k gthread -w 1 --threads 50 --timeout 120 --chdir command_center -b 0.0.0.0:${PORT:-8000} app:app"]
