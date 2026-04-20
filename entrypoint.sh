#!/usr/bin/env bash
# ============================================================
# lucebox-hub entrypoint
# Usage:
#   help           — show this message
#   megakernel     — run megakernel benchmark
#   dflash [ARGS]  — run dflash inference
#   download       — download required models via HF
#   shell          — drop into bash
# ============================================================
set -euo pipefail

cmd="${1:-help}"
shift || true   # remaining args forwarded to subcommand

case "$cmd" in

  # ── Help ────────────────────────────────────────────────────
  help)
    cat <<'EOF'

  ██╗     ██╗   ██╗ ██████╗███████╗██████╗  ██████╗ ██╗  ██╗
  ██║     ██║   ██║██╔════╝██╔════╝██╔══██╗██╔═══██╗╚██╗██╔╝
  ██║     ██║   ██║██║     █████╗  ██████╔╝██║   ██║ ╚███╔╝
  ██║     ██║   ██║██║     ██╔══╝  ██╔══██╗██║   ██║ ██╔██╗
  ███████╗╚██████╔╝╚██████╗███████╗██████╔╝╚██████╔╝██╔╝ ██╗
  ╚══════╝ ╚═════╝  ╚═════╝╚══════╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝

  RTX 3090 · sm_86 · CUDA 12.4

  Subcommands:
    help                  — this message
    download              — download Qwen3.5-27B-Q4_K_M + DFlash draft
    megakernel            — run megakernel benchmark (Qwen 3.5-0.8B)
    dflash [--prompt STR] — run DFlash 27B inference
    shell                 — interactive bash shell

  Quick start:
    docker compose run lucebox download
    docker compose run lucebox megakernel
    docker compose run lucebox dflash --prompt "def fibonacci(n):"

EOF
    ;;

  # ── Model download ──────────────────────────────────────────
  download)
    echo "==> Downloading Qwen3.5-27B-Q4_K_M GGUF target model (~14.9 GB)..."
    huggingface-cli download \
        unsloth/Qwen3.5-27B-GGUF \
        Qwen3.5-27B-Q4_K_M.gguf \
        --local-dir /workspace/lucebox-hub/dflash/models/

    echo ""
    echo "==> Downloading z-lab DFlash draft weights (~3.5 GB)..."
    huggingface-cli download \
        z-lab/Qwen3.5-27B-DFlash \
        model.safetensors \
        --local-dir /workspace/lucebox-hub/dflash/models/draft/

    echo ""
    echo "==> All models downloaded. Disk usage:"
    du -sh /workspace/lucebox-hub/dflash/models/
    ;;

  # ── Megakernel benchmark ────────────────────────────────────
  megakernel)
    echo "==> Running Megakernel benchmark (Qwen 3.5-0.8B, fused CUDA dispatch)..."
    cd /workspace/lucebox-hub/megakernel
    python final_bench.py "$@"
    ;;

  # ── DFlash 27B inference ────────────────────────────────────
  dflash)
    TARGET_GGUF="/workspace/lucebox-hub/dflash/models/Qwen3.5-27B-Q4_K_M.gguf"
    DRAFT_DIR="/workspace/lucebox-hub/dflash/models/draft/"

    if [[ ! -f "$TARGET_GGUF" ]]; then
      echo "ERROR: Target model not found at $TARGET_GGUF"
      echo "       Run:  docker compose run lucebox download"
      exit 1
    fi
    if [[ ! -f "${DRAFT_DIR}/model.safetensors" ]]; then
      echo "ERROR: Draft model not found at ${DRAFT_DIR}/model.safetensors"
      echo "       Run:  docker compose run lucebox download"
      exit 1
    fi

    echo "==> Running DFlash 27B (budget=22, RTX 3090 sweet spot)..."
    cd /workspace/lucebox-hub/dflash
    python3 scripts/run.py "$@"
    ;;

  # ── Interactive shell ────────────────────────────────────────
  shell)
    exec /bin/bash
    ;;

  # ── Pass-through for arbitrary commands ─────────────────────
  *)
    exec "$cmd" "$@"
    ;;

esac
