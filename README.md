# lucebox-hub · Docker Setup for RTX 3090

End-to-end Docker build for [lucebox-hub](https://github.com/Luce-Org/lucebox-hub), targeting the **RTX 3090 (Ampere sm_86)**. Covers both subprojects:

- **Megakernel** — fused single-dispatch CUDA kernel for Qwen 3.5-0.8B
- **DFlash 27B** — DFlash speculative decoding, Qwen 3.5-27B at ~130 tok/s on 24 GB VRAM

For an Unraid-first setup with stack-friendly paths, see [UNRAID.md](UNRAID.md).

---

## Prerequisites

| Requirement | Version |
|---|---|
| NVIDIA Driver | 525+ (for CUDA 12) |
| Docker Engine | 24+ |
| NVIDIA Container Toolkit | latest |
| Disk space | ~25 GB for models + image |

> **Your setup detected:** Driver 580.142 · CUDA 13.0 · RTX 3090 24 GB ✓  
> The image uses `nvidia/cuda:12.6.3-devel` as base — CUDA 13.0 Docker images aren't published yet, but your driver is fully backward-compatible with any CUDA 12.x binary.

Install the NVIDIA Container Toolkit if you haven't:
```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU is visible to Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## File layout

```
lucebox-hub-docker/
├── Dockerfile          ← CUDA 12.4 devel image, builds megakernel + dflash
├── docker-compose.yml  ← GPU passthrough, volumes, env
├── .env.example        ← Editable host-path + token template
├── entrypoint.sh       ← Subcommand router (download / megakernel / dflash / shell)
├── models/             ← Created automatically, holds GGUF + draft weights
└── hf_cache/           ← HuggingFace cache (avoids re-downloads)
```

---

## Quick Start

### 0. Configure host paths (recommended)

```bash
cp .env.example .env
```

On Unraid, edit `.env` and set absolute paths:

```bash
LUCEBOX_MODELS_DIR=/mnt/user/appdata/lucebox-hub/models
LUCEBOX_HF_CACHE_DIR=/mnt/user/appdata/lucebox-hub/hf_cache
HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
LUCEBOX_SERVER_HOST=0.0.0.0
LUCEBOX_SERVER_PORT=8000
LUCEBOX_BIND_IP=0.0.0.0
LUCEBOX_HOST_PORT=8000
DFLASH_SERVER_BUDGET=22
```

Port behavior:
- `LUCEBOX_SERVER_PORT` is the port the API listens on inside the container.
- `LUCEBOX_HOST_PORT` is the published host port on Unraid.
- `LUCEBOX_BIND_IP` controls which host interface is exposed (`0.0.0.0` for LAN, `127.0.0.1` for local-only).

### 1. Build the image

The build compiles the DFlash C++/CUDA engine (target `sm_86`) and installs the Python megakernel package. Takes ~10–15 min the first time.

```bash
docker compose build
```

Or with plain Docker:
```bash
docker build -t lucebox-hub:rtx3090 .
```

### 2. Download models (~18 GB total)

```bash
docker compose run --rm lucebox download
```

This downloads into your host path from `.env` (default `./models/`) so you only do it once.

### 3. Start persistent API server

```bash
docker compose up -d
```

Verify server health:

```bash
curl http://localhost:${LUCEBOX_HOST_PORT:-8000}/v1/models
```

Stop server:

```bash
docker compose down
```

### 4. Run Megakernel benchmark

Benchmarks the fused CUDA dispatch for Qwen 3.5-0.8B. Weights are streamed from HuggingFace automatically (no separate download needed).

```bash
docker compose run lucebox megakernel
```

Expected output (RTX 3090 @ 220W):

| Method | Prefill pp520 | Decode tg128 | tok/J |
|---|---|---|---|
| **Megakernel** | 37,800 | 413 | **1.87** |
| llama.cpp BF16 | 11,247 | 267 | 0.76 |

### 5. Run DFlash 27B inference

```bash
docker compose run lucebox dflash --prompt "def fibonacci(n):"
```

~130 tok/s on HumanEval prompts. 128K context fits in 24 GB (Q4_K_M target + BF16 draft + budget=22 DDTree + KV cache).

---

**Optional: Set Power Limit**

Your `nvidia-smi` shows a 225W cap. The project recommends 220W for best efficiency (power ceiling before compute ceiling = more tok/J). Drop it 5W with:

```bash
# On the HOST before starting container
sudo nvidia-smi -pl 220
```

---

## Interactive shell

```bash
docker compose run lucebox shell
# Inside container:
cd /workspace/lucebox-hub/dflash
python3 scripts/run.py --prompt "Explain transformers" --max_tokens 200
```

---

## Troubleshooting

**`CUDA error: no kernel image is available`**
→ The image is built for `sm_86`. Confirm you're on an RTX 3090 (or other Ampere GPU). RTX 30xx series = Ampere = sm_86.

**`Out of memory` during DFlash**
→ Make sure no other processes are using VRAM. Run `nvidia-smi` on host to check.

**`hf download` fails with 401/403**
→ Some models may require a HF account. Set `HUGGING_FACE_HUB_TOKEN` in the environment.

**Build fails on megakernel `pip install -e .`**
→ Likely a CUDA version mismatch in extension compilation. The Dockerfile uses PyTorch cu124 wheels against CUDA 12.6 devel — both are compatible with your 580.142 driver.
