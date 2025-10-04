# syntax=docker/dockerfile:1

FROM rust:1.76-slim AS rust-builder
WORKDIR /app
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY engine ./engine
COPY apps ./apps
COPY serving ./serving
RUN cargo build --release -p ledger-server
RUN cargo build --release --manifest-path serving/rust/Cargo.toml

FROM node:20-slim AS ui-builder
WORKDIR /web
COPY package.json package-lock.json ./
COPY web ./web
RUN npm ci && npm run build -w web/ui

FROM python:3.11-slim AS python-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /srv/foundation

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt && \
    pip install uvicorn

COPY . .
RUN python -m pip install -e ./model

COPY --from=rust-builder /app/target/release/ledger-server /usr/local/bin/ledger-server
COPY --from=rust-builder /app/target/release/serving /usr/local/bin/foundation-serving
COPY --from=ui-builder /web/web/ui/dist /srv/foundation/web/ui/dist

FROM python-base AS training
CMD ["python", "-m", "fm_train.trainer.run", "--config", "configs/train/default.yaml"]

FROM python-base AS serving
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -f http://localhost:8080/healthz || exit 1
CMD ["uvicorn", "serving.python.src.app:app", "--host", "0.0.0.0", "--port", "8080"]
