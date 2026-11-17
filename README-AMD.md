# rec-amd — ops_rec: AMD-tailored fork of recursive-oss/rec

Fork of https://github.com/recursive-oss/rec ("lightning-fast inference
engine", Python, Apache-2.0) at alexcasper/rec, tailored for **inference on
AMD hardware**.

## Mission

Port/tune rec for AMD inference targets:

- ROCm path for dusk-class GPUs (and Apocrita's AMD nodes if applicable)
- Vulkan fallback where ROCm is unavailable
- dawn's 8700G iGPU (RDNA3, 780M-class) as a smoke-test target
- Replace/abstract any CUDA-specific kernels, imports, or assumptions
- Benchmark against the dusk llama.cpp flint server (http://100.98.73.29:8080/v1)
  as the incumbent AMD serving stack

## Conventions

- Upstream: recursive-oss/rec (add tracking via `git fetch upstream`); our
  tailored work lives on branches off main; keep PRs upstreamable where the
  change is generic
- Repo-scoped project memory via entire-brain (`entire brain recall/remember`)
- No credentials in the repo; W&B/HF tokens stay in profile `.env`

## First tasks

1. Survey the codebase for CUDA/vendor-specific code paths and produce an
   AMD-port assessment (bead rec P1)
2. Verify the engine runs CPU-only on dawn as a baseline
3. ROCm/Vulkan device bring-up on dusk
