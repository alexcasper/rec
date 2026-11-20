#!/usr/bin/env bash
# rec-b3u: llama.cpp a37cc46 (upstream, "models : add Flint model") HIP build for gfx1201 on albion.
# Own tree (~/rec-ve/b3u/llama.cpp); the shared ~/llama.cpp belongs to another lane and is untouched.
# Toolchain = albion's known-good mixed recipe (~/llama.cpp/configure-mixed.sh): nix gcc-wrapper 14.2.1 for
# host C/C++ (the nix rocm clang 22 as host compiler clashes with glibc's posix_memalign decl), rocm clang++ for HIP.
set -eu
mkdir -p ~/rec-ve/b3u && cd ~/rec-ve/b3u
if [ ! -d llama.cpp/.git ]; then
  git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp llama.cpp
fi
cd llama.cpp && git checkout -q a37cc46 && git log --oneline -1
CMAKE=/nix/store/29ax4k0a83zhz43lb73cv610d95wdsx1-cmake-3.31.6/bin/cmake
NINJA=$(ls -d /nix/store/*-ninja-1.1[23]*/bin/ninja | grep -v drv | head -1)
export LD_LIBRARY_PATH=$(ls -d /nix/store/*-gcc-14.2.1.20250322-lib | head -1)/lib:$(ls -d /nix/store/*-zstd-1.5.7 | head -1)/lib:$(ls -d /nix/store/*-zlib-1.3.2 | head -1)/lib
GW=/nix/store/a0d7m3zn9p2dfa1h7ag9h2wzzr2w25sn-gcc-wrapper-14.2.1.20250322
rm -rf build-hip
export HIPCXX=/opt/rocm/llvm/bin/clang++ HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm
$CMAKE -S . -B build-hip -G Ninja -DCMAKE_MAKE_PROGRAM=$NINJA \
  -DCMAKE_C_COMPILER=$GW/bin/gcc -DCMAKE_CXX_COMPILER=$GW/bin/g++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm;/nix/store/glgfmgk4ds85b3lgwl5c6112xm9h6yvc-rocm-comgr-22.0.0-rocm;/nix/store/4pkzqs35xgas5np8010kqzis0zx39kzj-hipblas-common-7.2.3" \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON -DGGML_CUDA_FA=ON \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > cmake.log 2>&1
$CMAKE --build build-hip -j 14 --target llama-bench llama-quantize llama-cli llama-server > build.log 2>&1
ls -la build-hip/bin/ | grep -E "llama-(bench|quantize|cli|server)$"
echo BUILD_OK
