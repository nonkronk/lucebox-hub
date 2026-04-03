<p align="center">
  <img src="hero.png" width="600" />
</p>

# Megakernel

Single CUDA kernel for the entire forward pass of Qwen 3.5-0.8B. 24 layers, one launch.

## Results (RTX 3090)

| | Prefill | Decode |
|---|---|---|
| **Megakernel** | **37,800 tok/s** | **413 tok/s** |
| llama.cpp BF16 | 11,247 | 267 |
| PyTorch HF | 7,578 | 108 |
| M5 Max | - | 229 |

At 220W: 411 tok/s, 1.87 tok/J. Matching M5 Max efficiency at 1.8x the speed.

## Run

```bash
pip install -e .
python bench_pp_tg.py
```

Needs: NVIDIA GPU (tested RTX 3090), CUDA 12+, PyTorch 2.0+

## Files

| File | What |
|---|---|
| `kernel.cu` | Decode megakernel (BF16, all 24 layers) |
| `prefill.cu` | Prefill kernels + cuBLAS |
| `torch_bindings.cpp` | PyTorch bindings |
| `model.py` | Weight loading, Decoder |
| `setup.py` | Build |
| `bench_pp_tg.py` | Benchmark |

## More

[Blog post](https://lucebox.com/blog/megakernel) - [lucebox.com](https://lucebox.com)

MIT License
