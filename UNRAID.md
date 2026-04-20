# Unraid Deployment Guide

This guide is tuned for Unraid hosts and keeps all large artifacts in appdata paths.

## Prerequisites

- Unraid with NVIDIA GPU support enabled.
- Docker service running on Unraid.
- An RTX 3090 or other Ampere GPU for the default sm_86 build.
- At least 25 GB free disk for image + model files.

## 1. Configure environment file

From your repo directory:

```bash
cd /mnt/primary_cache/appdata/llama_cpp/lucebox-hub
cp .env.example .env
```

Edit .env for persistent Unraid paths:

```bash
LUCEBOX_MODELS_DIR=/mnt/user/appdata/lucebox-hub/models
LUCEBOX_HF_CACHE_DIR=/mnt/user/appdata/lucebox-hub/hf_cache
HUGGING_FACE_HUB_TOKEN=
LUCEBOX_SERVER_HOST=0.0.0.0
LUCEBOX_SERVER_PORT=8000
LUCEBOX_BIND_IP=0.0.0.0
LUCEBOX_HOST_PORT=8000
DFLASH_SERVER_BUDGET=22
```

Set HUGGING_FACE_HUB_TOKEN only if your model access requires login.

Port behavior:
- `LUCEBOX_SERVER_PORT` is the API listen port inside the container.
- `LUCEBOX_HOST_PORT` is the published host port on Unraid.
- `LUCEBOX_BIND_IP` controls exposure (`0.0.0.0` for LAN, `127.0.0.1` for local-only).

## 2. Build image

Use an absolute compose path so the command works even if your shell cwd changes:

```bash
docker compose -f /mnt/primary_cache/appdata/llama_cpp/lucebox-hub/docker-compose.yml build
```

## 3. Download models

```bash
docker compose -f /mnt/primary_cache/appdata/llama_cpp/lucebox-hub/docker-compose.yml run --rm lucebox download
```

This uses hf download inside the container and writes models to LUCEBOX_MODELS_DIR.

## 4. Start persistent API service

```bash
docker compose -f /mnt/primary_cache/appdata/llama_cpp/lucebox-hub/docker-compose.yml up -d
```

Check status and logs:

```bash
docker compose -f /mnt/primary_cache/appdata/llama_cpp/lucebox-hub/docker-compose.yml ps
docker compose -f /mnt/primary_cache/appdata/llama_cpp/lucebox-hub/docker-compose.yml logs -f lucebox
```

Verify API:

```bash
curl http://localhost:${LUCEBOX_HOST_PORT:-8000}/v1/models
```

## 5. Run DFlash one-shot inference

```bash
docker compose -f /mnt/primary_cache/appdata/llama_cpp/lucebox-hub/docker-compose.yml run --rm lucebox dflash --prompt "def fibonacci(n):"
```

## 6. Run megakernel benchmark

```bash
docker compose -f /mnt/primary_cache/appdata/llama_cpp/lucebox-hub/docker-compose.yml run --rm lucebox megakernel
```

## 7. Unraid UI flow (Compose Manager)

1. Open Docker tab, then Compose Manager.
2. Open the lucebox-hub stack.
3. Confirm .env has your absolute host paths.
4. Click Compose Up.
5. Open Container Logs for lucebox and wait for the line showing the server URL.
6. Test from Unraid terminal: curl http://localhost:${LUCEBOX_HOST_PORT:-8000}/v1/models.
7. For auto-start on reboot, enable autostart for the stack/container in Unraid.

## 8. Optional operations

Open shell in container:

```bash
docker compose -f /mnt/primary_cache/appdata/llama_cpp/lucebox-hub/docker-compose.yml run --rm lucebox shell
```

Stop and remove stack resources:

```bash
docker compose -f /mnt/primary_cache/appdata/llama_cpp/lucebox-hub/docker-compose.yml down
```

## Unraid notes

- If GPU is not visible, verify NVIDIA support on host first:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

- If download fails with 401/403, set HUGGING_FACE_HUB_TOKEN in .env and rerun download.
- If you use Unraid Compose Manager, point it to this repo and use docker-compose.yml plus .env.
