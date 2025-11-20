# syntax=docker/dockerfile:1.7

FROM rust:1.88-bullseye AS wheel-builder
ARG PROTOC_VERSION=28.3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    curl \
    pkg-config \
    libssl-dev \
    build-essential \
    libprotobuf-dev \
    protobuf-compiler \
    unzip \
    && rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
        x86_64|amd64) protoc_zip="protoc-${PROTOC_VERSION}-linux-x86_64.zip" ;; \
        arm64|aarch64) protoc_zip="protoc-${PROTOC_VERSION}-linux-aarch_64.zip" ;; \
        *) echo "Unsupported architecture: ${arch}" && exit 1 ;; \
    esac; \
    curl -LO "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/${protoc_zip}"; \
    mkdir -p /tmp/protoc && unzip -q "${protoc_zip}" -d /tmp/protoc; \
    mv /tmp/protoc/bin/protoc /usr/local/bin/protoc; \
    chmod +x /usr/local/bin/protoc; \
    cp -R /tmp/protoc/include/* /usr/local/include/; \
    rm -rf /tmp/protoc "${protoc_zip}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /repo
COPY . /repo
RUN uv run scripts/build_wheel.py --out-dir target/wheels

FROM python:3.12-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY example_app /app

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY --from=wheel-builder /repo/target/wheels /tmp/wheels
RUN set -eux; wheel=$(ls /tmp/wheels/rappel-*.whl | head -n 1); uv pip install --no-cache-dir "$wheel"
RUN uv pip install --no-cache-dir ".[tests]"

RUN chmod +x scripts/start-daemons.sh

ENV CARABINER_HTTP_PORT=24117 \
    CARABINER_GRPC_PORT=24118

CMD ["uvicorn", "example_app.web:app", "--host", "0.0.0.0", "--port", "8000"]
