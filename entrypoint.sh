#!/usr/bin/env bash
# ============================================================
# lucebox-hub entrypoint
# Usage:
#   help           — show this message
#   megakernel     — run megakernel benchmark
#   dflash [ARGS]  — run dflash inference
#   serve [ARGS]   — run persistent OpenAI-compatible server
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
    serve                 — run persistent OpenAI-compatible API (:8000)
    shell                 — interactive bash shell

  Quick start:
    docker compose run lucebox download
    docker compose up -d
    curl http://localhost:8000/v1/models

EOF
    ;;

  # ── Model download ──────────────────────────────────────────
  download)
    if ! command -v hf >/dev/null 2>&1; then
      echo "ERROR: Hugging Face CLI 'hf' not found in container PATH"
      echo "       Install with: pip install \"huggingface_hub[cli]\""
      exit 1
    fi

    HF_TOKEN_ARGS=()
    if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
      HF_TOKEN_ARGS+=(--token "${HUGGING_FACE_HUB_TOKEN}")
    fi

    echo "==> Downloading Qwen3.5-27B-Q4_K_M GGUF target model (~14.9 GB)..."
    hf download \
        unsloth/Qwen3.5-27B-GGUF \
        Qwen3.5-27B-Q4_K_M.gguf \
        "${HF_TOKEN_ARGS[@]}" \
        --local-dir /workspace/lucebox-hub/dflash/models/

    echo ""
    echo "==> Downloading z-lab DFlash draft weights (~3.5 GB)..."
    hf download \
        z-lab/Qwen3.5-27B-DFlash \
        model.safetensors \
        "${HF_TOKEN_ARGS[@]}" \
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

  # ── Persistent OpenAI-compatible API server ────────────────
  serve)
    TARGET_GGUF="/workspace/lucebox-hub/dflash/models/Qwen3.5-27B-Q4_K_M.gguf"
    DRAFT_SAFETENSORS="/workspace/lucebox-hub/dflash/models/draft/model.safetensors"
    SERVER_HOST="${LUCEBOX_SERVER_HOST:-0.0.0.0}"
    SERVER_PORT="${LUCEBOX_SERVER_PORT:-8000}"
    SERVER_BUDGET="${DFLASH_SERVER_BUDGET:-22}"

    if [[ ! -f "$TARGET_GGUF" ]]; then
      echo "ERROR: Target model not found at $TARGET_GGUF"
      echo "       Run once: docker compose run --rm lucebox download"
      exit 1
    fi
    if [[ ! -f "$DRAFT_SAFETENSORS" ]]; then
      echo "ERROR: Draft model not found at $DRAFT_SAFETENSORS"
      echo "       Run once: docker compose run --rm lucebox download"
      exit 1
    fi

    echo "==> Starting persistent DFlash OpenAI API on ${SERVER_HOST}:${SERVER_PORT}"
    cd /workspace/lucebox-hub/dflash
    exec python3 scripts/server.py \
      --host "$SERVER_HOST" \
      --port "$SERVER_PORT" \
      --target "$TARGET_GGUF" \
      --draft "$DRAFT_SAFETENSORS" \
      --budget "$SERVER_BUDGET" \
      "$@"
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
