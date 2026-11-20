// rec-6ob lab: load-path variants of the fused INT4-affine gemv (flint_q4_gemv_op), standalone.
// Includes the run-of-record source for dot16 / load16_f / simd_sum / tofloat / tobf, so every variant
// uses the exact same per-element math. Every variant keeps each row's fp32 operation order identical
// to flint_q4_gemv_op<K> (lane l owns elements [16l, 16l+16) of every 512-block; per-row sum
// a += dot16(blk0); a += dot16(blk1); ...; then simd_sum butterfly). Only scheduling changes,
// so outputs must be BYTE-identical to the production kernel (checked by q4_lab.py).
//
//   V1 inline      : flint_q4_gemv_op body, __attribute__((always_inline)) (global loads, no call/spill)
//   V2 batched     : V1 + unguarded fast path for full row-quads: all 4 rows' weight/scale/bias loads
//                    are issued before the first use (the per-row `if (row < n)` serialized them)
//   V3 prefetch    : V2 + next row-quad's packed weights prefetched into registers during compute
//   V4 rows8       : V2 with 8 rows per warp iteration
//   STREAM         : pure read of N*K/2 bytes (uint4 per lane, grid-stride) = bandwidth ceiling
#include "flint_mid_megakernel.hip"

#define FI __device__ __attribute__((always_inline)) inline

template <int K>
FI void stage_x(const float* x, int lane, float (&v)[K / 512][16], float (&sm)[K / 512]) {
    #pragma unroll
    for (int blk = 0; blk < K / 512; ++blk) sm[blk] = load16_f(x + blk * 512 + lane * 16, v[blk]);
}

template <int K>
FI void store_row(float val, int row, int lane, bf16* out_bf, float* out_f) {
    if (lane == 0) {
        if (out_bf) out_bf[row] = tobf(val);
        if (out_f) out_f[row] = val;
    }
}

// V1: production op, inlined. Identical source to flint_q4_gemv_op<K, false> (both branches).
template <int K>
FI void gemv_v1(const uint8_t* __restrict__ w, const bf16* __restrict__ s, const bf16* __restrict__ b,
                const float* __restrict__ x, bf16* __restrict__ out_bf, float* __restrict__ out_f,
                int n, int lane, int sg, int groups) {
    float v[K / 512][16], sm[K / 512];
    stage_x<K>(x, lane, v, sm);
    for (int first = sg * 4; first < n; first += groups * 4) {
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int row = first + r;
            if (row < n) {
                const uint8_t* wr = w + (size_t)row * (K / 2);
                const bf16* sr = s + (size_t)row * (K / 64);
                const bf16* br = b + (size_t)row * (K / 64);
                #pragma unroll
                for (int blk = 0; blk < K / 512; ++blk) {
                    int sc = blk * 512 + lane * 16;
                    a[r] += dot16(wr + (sc >> 1), v[blk], tofloat(sr[sc >> 6]), tofloat(br[sc >> 6]), sm[blk]);
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            float val = simd_sum(a[r]);
            if (first + r < n) store_row<K>(val, first + r, lane, out_bf, out_f);
        }
    }
}

// Raw operands for one row: 8 bytes of packed weights per 512-block + that block's scale/bias.
template <int K>
struct RowOps {
    uint2 wq[K / 512];
    float sc[K / 512], bi[K / 512];
};

template <int K>
FI void load_row(const uint8_t* w, const bf16* s, const bf16* b, int row, int lane, RowOps<K>& o) {
    const uint8_t* wr = w + (size_t)row * (K / 2);
    const bf16* sr = s + (size_t)row * (K / 64);
    const bf16* br = b + (size_t)row * (K / 64);
    #pragma unroll
    for (int blk = 0; blk < K / 512; ++blk) {
        int scl = blk * 512 + lane * 16;
        o.wq[blk] = *(const uint2*)(wr + (scl >> 1));
        o.sc[blk] = tofloat(sr[scl >> 6]);
        o.bi[blk] = tofloat(br[scl >> 6]);
    }
}

template <int K>
FI float dot_row(const RowOps<K>& o, float (&v)[K / 512][16], float (&sm)[K / 512]) {
    float a = 0.0f;
    #pragma unroll
    for (int blk = 0; blk < K / 512; ++blk) a += dot16((const uint8_t*)&o.wq[blk], v[blk], o.sc[blk], o.bi[blk], sm[blk]);
    return a;
}

// V2 (ROWS=4) / V4 (ROWS=8): full row-groups take an unguarded path with all loads issued first;
// the tail group falls back to the guarded V1 form. PREFETCH (V3) double-buffers the next group.
template <int K, int ROWS, bool PREFETCH>
FI void gemv_batched(const uint8_t* __restrict__ w, const bf16* __restrict__ s, const bf16* __restrict__ b,
                     const float* __restrict__ x, bf16* __restrict__ out_bf, float* __restrict__ out_f,
                     int n, int lane, int sg, int groups) {
    float v[K / 512][16], sm[K / 512];
    stage_x<K>(x, lane, v, sm);
    int first = sg * ROWS;
    const int step = groups * ROWS;
    RowOps<K> cur[ROWS];
    if (PREFETCH && first + ROWS <= n) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) load_row<K>(w, s, b, first + r, lane, cur[r]);
    }
    for (; first + ROWS <= n; first += step) {
        RowOps<K> ops[ROWS];
        if (PREFETCH) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) ops[r] = cur[r];
            int nxt = first + step;
            if (nxt + ROWS <= n) {
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) load_row<K>(w, s, b, nxt + r, lane, cur[r]);
            }
        } else {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) load_row<K>(w, s, b, first + r, lane, ops[r]);
        }
        float a[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) a[r] = dot_row<K>(ops[r], v, sm);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) store_row<K>(simd_sum(a[r]), first + r, lane, out_bf, out_f);
    }
    if (first < n) {  // partial tail group (n % ROWS != 0 for this warp's last group)
        float a[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            a[r] = 0.0f;
            if (first + r < n) {
                RowOps<K> o;
                load_row<K>(w, s, b, first + r, lane, o);
                a[r] = dot_row<K>(o, v, sm);
            }
        }
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float val = simd_sum(a[r]);
            if (first + r < n) store_row<K>(val, first + r, lane, out_bf, out_f);
        }
    }
}

template <int K, int VARIANT>
__global__ __launch_bounds__(256) void lab_kernel(const uint8_t* w, const bf16* s, const bf16* b,
                                                  const bf16* x, bf16* out_bf, float* out_f, int n) {
    int tid = threadIdx.x, lane = tid & 31;
    int sg = blockIdx.x * (blockDim.x >> 5) + (tid >> 5);
    int groups = gridDim.x * (blockDim.x >> 5);
    __shared__ float sh_x[K];
    for (int i = tid; i < K; i += 256) sh_x[i] = tofloat(x[i]);
    __syncthreads();
    if constexpr (VARIANT == 1) gemv_v1<K>(w, s, b, sh_x, out_bf, out_f, n, lane, sg, groups);
    if constexpr (VARIANT == 2) gemv_batched<K, 4, false>(w, s, b, sh_x, out_bf, out_f, n, lane, sg, groups);
    if constexpr (VARIANT == 3) gemv_batched<K, 4, true>(w, s, b, sh_x, out_bf, out_f, n, lane, sg, groups);
    if constexpr (VARIANT == 4) gemv_batched<K, 8, false>(w, s, b, sh_x, out_bf, out_f, n, lane, sg, groups);
}

// Pure streaming read: every byte of the weight array + scales + biases once, uint4 per lane.
__global__ __launch_bounds__(256) void stream_kernel(const uint4* p, size_t n16, float* sink) {
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
        uint4 q = p[i];
        acc ^= q.x ^ q.y ^ q.z ^ q.w;
    }
    if (acc == 0x9e3779b9u) sink[0] = 1.0f;  // keep the loads alive
}

template <int K>
static void launch_lab_k(int variant, int grid, const uint8_t* w, const bf16* s, const bf16* b, const bf16* x,
                         float* o, int n) {
    switch (variant) {
        case 1: hipLaunchKernelGGL((lab_kernel<K, 1>), dim3(grid), dim3(256), 0, 0, w, s, b, x, nullptr, o, n); break;
        case 2: hipLaunchKernelGGL((lab_kernel<K, 2>), dim3(grid), dim3(256), 0, 0, w, s, b, x, nullptr, o, n); break;
        case 3: hipLaunchKernelGGL((lab_kernel<K, 3>), dim3(grid), dim3(256), 0, 0, w, s, b, x, nullptr, o, n); break;
        case 4: hipLaunchKernelGGL((lab_kernel<K, 4>), dim3(grid), dim3(256), 0, 0, w, s, b, x, nullptr, o, n); break;
    }
}

static void launch_lab(int variant, int k, int grid, const uint8_t* w, const bf16* s, const bf16* b,
                       const bf16* x, float* o, int n) {
    if (k == 2048) launch_lab_k<2048>(variant, grid, w, s, b, x, o, n);
    else if (k == 4096) launch_lab_k<4096>(variant, grid, w, s, b, x, o, n);
    else if (k == 1024) launch_lab_k<1024>(variant, grid, w, s, b, x, o, n);
    else if (k == 512) launch_lab_k<512>(variant, grid, w, s, b, x, o, n);
}


// ---- Global-x variants (mirror the megakernel: x is an fp32 workspace buffer in global memory) ----
// V11: production block-outer order (x slice reloaded per block) but each block's four row loads are
//      issued together on full row-quads (no per-row guards); tail falls back to the guarded form.
// V12: V11 reading x from an LDS copy staged once per workgroup.
template <int K>
FI void gemv_generic_batched(const uint8_t* __restrict__ w, const bf16* __restrict__ s, const bf16* __restrict__ b,
                             const float* __restrict__ x, float* __restrict__ out_f, int n, int lane, int sg,
                             int groups) {
    for (int first = sg * 4; first < n; first += groups * 4) {
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const bool full = first + 4 <= n;
        #pragma unroll
        for (int block = 0; block < K; block += 512) {
            int scalar = block + lane * 16;
            float v[16];
            float sum = load16_f(x + scalar, v);
            if (full) {
                uint2 wq[4];
                float sc[4], bi[4];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int gi = (first + r) * (K / 64) + scalar / 64;
                    wq[r] = *(const uint2*)(w + (size_t)(first + r) * (K / 2) + scalar / 2);
                    sc[r] = tofloat(s[gi]);
                    bi[r] = tofloat(b[gi]);
                }
                #pragma unroll
                for (int r = 0; r < 4; ++r) a[r] += dot16((const uint8_t*)&wq[r], v, sc[r], bi[r], sum);
            } else {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int row = first + r;
                    if (row < n) {
                        int gi = row * (K / 64) + scalar / 64;
                        a[r] += dot16(w + (size_t)row * (K / 2) + scalar / 2, v, tofloat(s[gi]), tofloat(b[gi]), sum);
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            float val = simd_sum(a[r]);
            if (lane == 0 && first + r < n) out_f[first + r] = val;
        }
    }
}

template <int K, int VARIANT>
__global__ __launch_bounds__(256) void lab_kernel_gx(const uint8_t* w, const bf16* s, const bf16* b,
                                                     const float* xg, float* out_f, int n) {
    int tid = threadIdx.x, lane = tid & 31;
    int sg = blockIdx.x * (blockDim.x >> 5) + (tid >> 5);
    int groups = gridDim.x * (blockDim.x >> 5);
    if constexpr (VARIANT == 10) {
        flint_q4_gemv_op<K>(w, s, b, xg, nullptr, out_f, n, lane, sg, groups);
    } else if constexpr (VARIANT == 11) {
        gemv_generic_batched<K>(w, s, b, xg, out_f, n, lane, sg, groups);
    } else {
        __shared__ float sh[K];
        for (int i = tid; i < K; i += 256) sh[i] = xg[i];
        __syncthreads();
        gemv_generic_batched<K>(w, s, b, sh, out_f, n, lane, sg, groups);
    }
}

template <int K>
static void launch_gx_k(int variant, int grid, const uint8_t* w, const bf16* s, const bf16* b, const float* x,
                        float* o, int n) {
    switch (variant) {
        case 10: hipLaunchKernelGGL((lab_kernel_gx<K, 10>), dim3(grid), dim3(256), 0, 0, w, s, b, x, o, n); break;
        case 11: hipLaunchKernelGGL((lab_kernel_gx<K, 11>), dim3(grid), dim3(256), 0, 0, w, s, b, x, o, n); break;
        case 12: hipLaunchKernelGGL((lab_kernel_gx<K, 12>), dim3(grid), dim3(256), 0, 0, w, s, b, x, o, n); break;
    }
}

static void launch_gx(int variant, int k, int grid, const uint8_t* w, const bf16* s, const bf16* b, const float* x,
                      float* o, int n) {
    if (k == 2048) launch_gx_k<2048>(variant, grid, w, s, b, x, o, n);
    else if (k == 4096) launch_gx_k<4096>(variant, grid, w, s, b, x, o, n);
    else if (k == 1024) launch_gx_k<1024>(variant, grid, w, s, b, x, o, n);
    else if (k == 512) launch_gx_k<512>(variant, grid, w, s, b, x, o, n);
}

static int default_grid(int n, int rows) {
    int warps = (n + rows - 1) / rows;
    int g = (warps + 7) / 8;
    return g > 256 ? 256 : (g < 1 ? 1 : g);
}

// variant 0..4 timed on device buffers; variant 9 = stream. Returns ms/iter; outputs left in d_out_f.
extern "C" int lab_run(int variant, const void* d_w, const void* d_s, const void* d_b, const void* d_x,
                       float* d_out_f, int n, int k, int grid, int warmup, int iters, float* out_ms) {
    if (grid < 1) grid = default_grid(n, variant == 4 ? 8 : 4);
    auto launch = [&]() {
        if (variant == 9) {
            size_t bytes = (size_t)n * (k / 2);
            hipLaunchKernelGGL(stream_kernel, dim3(grid), dim3(256), 0, 0, (const uint4*)d_w, bytes / 16, d_out_f);
        } else if (variant == 0) {
            flint_q4_gemv_device(d_w, d_s, d_b, d_x, nullptr, d_out_f, n, k, grid);
        } else if (variant >= 10) {  // d_x is an fp32 [K] buffer for the global-x variants
            launch_gx(variant, k, grid, (const uint8_t*)d_w, (const bf16*)d_s, (const bf16*)d_b, (const float*)d_x,
                      d_out_f, n);
        } else {
            launch_lab(variant, k, grid, (const uint8_t*)d_w, (const bf16*)d_s, (const bf16*)d_b,
                       (const bf16*)d_x, d_out_f, n);
        }
        return (int)hipGetLastError();
    };
    for (int i = 0; i < warmup; ++i)
        if (int rc = launch()) return rc;
    if (hipDeviceSynchronize() != hipSuccess) return -3;
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipEventRecord(e0, 0);
    for (int i = 0; i < iters; ++i)
        if (int rc = launch()) return rc;
    hipEventRecord(e1, 0);
    hipEventSynchronize(e1);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, e0, e1);
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    *out_ms = ms / (float)iters;
    return (int)hipGetLastError();
}

extern "C" void* lab_alloc(size_t bytes) { void* p = nullptr; return hipMalloc(&p, bytes) == hipSuccess ? p : nullptr; }
extern "C" int lab_free(void* p) { return (int)hipFree(p); }
extern "C" int lab_h2d(void* d, const void* h, size_t bytes) { return (int)hipMemcpy(d, h, bytes, hipMemcpyHostToDevice); }
extern "C" int lab_d2h(void* h, const void* d, size_t bytes) { return (int)hipMemcpy(h, d, bytes, hipMemcpyDeviceToHost); }
extern "C" int lab_memset(void* d, int v, size_t bytes) { return (int)hipMemset(d, v, bytes); }
