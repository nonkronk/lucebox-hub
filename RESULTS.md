# Qwen3.5-0.8B BF16 Megakernel — Benchmark Results

## Hardware

| Machine | GPU/Chip | Memory |
|---------|----------|--------|
| Lucebox | NVIDIA RTX 3090 (24GB VRAM) | 32GB RAM |
| MacBook Pro | Apple M5 Max | 36GB Unified |

## RTX 3090: pp520 tg128

| Method | pp520 (tok/s) | tg128 (tok/s) | Completion |
|--------|:---:|:---:|---|
| **Our megakernel** | **37,800** (warmed) | **413** | ✓ "Paris." |
| llama.cpp BF16 | 11,247 | 267 | ✓ (--reasoning-budget 0) |
| PyTorch HF | 7,578 | 108 | ✓ Same as ours |

### RTX 3090 Decode Speedups

| | vs llama.cpp | vs PyTorch |
|---|:---:|:---:|
| **Decode (tg128)** | **1.55x** | **3.8x** |

## Apple M5 Max: Decode

| Method | tg (tok/s) |
|--------|:---:|
| LM Studio (llama.cpp) BF16 | 229 |

## RTX 3090: Power Efficiency (DVFS)

Megakernel decode at different power limits:

| Power Limit | Clock | Actual Draw | tok/s | tok/J | vs Stock |
|---|---|---|---|---|---|
| 420W (stock) | 1980 MHz | 314W | 433 | 1.38 | baseline |
| 300W | 1935 MHz | 299W | 432 | 1.44 | 99.8% speed, 5% less power |
| **220W** | **1635 MHz** | **220W** | **411** | **1.87** | **95% speed, 30% less power** |
| 150W | 405 MHz | 150W | 194 | 1.29 | too aggressive |

**Sweet spot: 220W — 1.87 tok/J, 35% more efficient than stock, only 5% speed loss.**

## Key Findings

- **Decode:** Our persistent megakernel is 1.55x faster than llama.cpp on RTX 3090. Single kernel launch processes all 24 layers with zero inter-layer overhead.
- **Prefill:** cuBLAS tensor cores are 3.4x faster than llama.cpp's custom `mmf` BF16 GEMM kernel (when warmed). Cold start is comparable.
- **Architecture:** Qwen3.5-0.8B uses a hybrid DeltaNet + Full Attention architecture that most frameworks (MLX, vLLM) don't natively support yet. Only llama.cpp and our engine have custom CUDA kernels for it.
- **Correctness:** Prefill → decode handoff verified. Output matches PyTorch HuggingFace exactly.

## Architecture

- **Decode:** Single persistent megakernel (82 blocks × 512 threads)
  - bf16 weights × bf16 activations, f32 accumulation
  - DeltaNet: warp-cooperative recurrence with f32 state
  - Full Attention: causal decode with online softmax + sigmoid gate
  - One kernel launch for all 24 layers

- **Prefill:** cuBLAS bf16 GEMM + standalone CUDA kernels
  - cuBLAS tensor core GEMM for all projections
  - Standalone DeltaNet recurrence kernel (state-in-registers, f32 state)
  - Standalone causal attention kernel

- **Weights:** All bf16 (1.5GB), loaded from HuggingFace directly
- **State handoff:** Seamless prefill → decode (same f32 DeltaNet state + bf16 KV cache)

## Files

- `kernel.cu` — decode megakernel
- `prefill.cu` — cuBLAS prefill orchestrator + standalone kernels
- `torch_bindings.cpp` — PyTorch C++ bindings
- `model.py` — weight loading + Decoder class
- `setup.py` — build config
- `final_bench.py` — pp520 tg128 benchmark
- `benchmark_bars.png` — comparison chart
- `decode_hero.png` — decode speed chart
