# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PORT=7860 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

RUN --mount=type=cache,id=lab7-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=lab7-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-dev \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0

COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,id=lab7-pip-cache,target=/root/.cache/pip,sharing=locked \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install -r /app/requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"

EXPOSE 7860
CMD ["python3", "app.py"]
