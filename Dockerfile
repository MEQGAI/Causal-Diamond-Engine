# syntax=docker/dockerfile:1

FROM rust:1.76-slim AS rust-builder
WORKDIR /app
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY engine ./engine
COPY apps ./apps
RUN cargo build --release -p ledger-server

FROM node:20-slim AS ui-builder
WORKDIR /web
COPY package.json package-lock.json ./
COPY web ./web
RUN npm ci && npm run build -w web/ui

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /srv/ledger

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copy sources
COPY . .
COPY --from=rust-builder /app/target/release/ledger-server /usr/local/bin/ledger-server
COPY --from=ui-builder /web/web/ui/dist /srv/ledger/web/ui/dist

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -f http://localhost:8080/healthz || exit 1

CMD ["ledger-server"]
