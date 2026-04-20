# ============================================================
# lucebox-hub — RTX 3090 (Ampere sm_86) Docker image
# Builds both megakernel and dflash 27B from source.
# Models are NOT baked in — mount them via ./models volume.
# ============================================================

# CUDA 12.6 devel image — nvcc + cuBLAS headers + runtime.
# Your driver is 580.142 (CUDA 13.0). NVIDIA drivers are forward-compatible:
# a CUDA 13.0 driver runs any CUDA 12.x binary without changes.
# CUDA 13.0 Docker images are not yet published on Docker Hub (as of Apr 2026),
# so 12.6 is the highest available devel image and is fully supported by your driver.
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# ── Labels ───────────────────────────────────────────────────
LABEL org.opencontainers.image.title="lucebox-hub"
LABEL org.opencontainers.image.description="Hand-tuned LLM inference for RTX 3090"
LABEL org.opencontainers.image.source="https://github.com/Luce-Org/lucebox-hub"

# ── Environment ──────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_ARCHITECTURES=86
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Hugging Face cache → inside the container; map to host via volume
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        curl \
        wget \
        ca-certificates \
        build-essential \
        ninja-build \
        cmake \
        python3 \
        python3-pip \
        python3-dev \
        libopenblas-dev \
        # nice-to-haves for interactive use
        htop \
        vim \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 → python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ── Python base ──────────────────────────────────────────────
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch 2.5 with CUDA 12.4 wheels — best match for CUDA 12.6 devel image.
# Runs fine under your 580.142 / CUDA 13.0 driver (backward compatible).
RUN pip3 install --no-cache-dir \
        torch==2.5.1 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

# Common Python deps used by both subprojects
RUN pip3 install --no-cache-dir \
        transformers \
        huggingface_hub \
        "huggingface_hub[cli]" \
        accelerate \
        safetensors \
        sentencepiece \
        einops \
        numpy \
        tqdm \
        pytest

# ── Clone repo (with submodules) ─────────────────────────────
WORKDIR /workspace
RUN git clone --recurse-submodules \
        https://github.com/Luce-Org/lucebox-hub.git \
        lucebox-hub

# ── Build: Megakernel ────────────────────────────────────────
# setup.py imports torch at build time — disable pip's isolated build
# subprocess so it can find the torch we installed in the previous layer.
WORKDIR /workspace/lucebox-hub/megakernel
RUN pip3 install --no-cache-dir --no-build-isolation -e .

# ── Build: DFlash 27B ────────────────────────────────────────
WORKDIR /workspace/lucebox-hub/dflash
# libcuda.so.1 is a host driver library — not present inside the build container.
# The CUDA devel image ships a linker stub at /usr/local/cuda/lib64/stubs/libcuda.so
# Create the .so.1 symlink the linker expects, point CMake at the stubs dir,
# then at runtime the real libcuda.so.1 from the host driver is used instead.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
           /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && cmake -B build -S . \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_FLAGS="-arch=sm_86 --use_fast_math" \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
    && cmake --build build --target test_dflash -j"$(nproc)" \
    && rm /usr/local/cuda/lib64/stubs/libcuda.so.1

# ── Model directory (populated at runtime via volume) ────────
RUN mkdir -p /workspace/lucebox-hub/dflash/models/draft

# ── Entrypoint script ────────────────────────────────────────
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace/lucebox-hub
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["help"]