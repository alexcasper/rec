// Flint megakernel — HIP phase-1 port of the Metal device functions
// (rec-tan). Literal translation of rec_mlx/kernels/megakernel.py math:
//   simd_sum (Apple simdgroup 32 -> RDNA wave32 butterfly shuffle)
//   load16/dot16/q4   INT4-affine quantized matmul with x-scaling bias trick
//   rms               RMSNorm, 4 elements per thread, warp+block reduction
//   head_norm         per-head RMSNorm + NeoX-half RoPE (pair, pair+half)
//   attention_partial attention with per-head SINKS as streaming-softmax
//                     initial hi/z (rec's softmax_offset semantics)
// Validated against torch references on the tiny Flint shapes
// (hidden 2048 for q4/rms as in real Flint; attention HEADS=8, head_dim 128).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

typedef __hip_bfloat16 bf16;

__device__ inline float tofloat(bf16 v) { return __bfloat162float(v); }
__device__ inline bf16 tobf(float v) { return __float2bfloat16(v); }

// ---- simd_sum: 32-lane butterfly (Apple simd_sum == RDNA wave32) ----
__device__ inline float simd_sum(float v) {
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        v += __shfl_xor_sync(0xffffffffffffffffull, v, off);
    return v;
}

// ---- quantized INT4-affine matmul (megakernel q4 / dot16 / load16) ----
struct Matrix {
    const uint8_t* w;   // packed nibbles, row-major: row * (K/2) bytes
    const bf16* s;      // scales [n, K/64]
    const bf16* b;      // biases [n, K/64]
    int n, k;
};

__device__ inline float load16(const bf16* x, float* v) {
    float sum = 0.0f;
    for (uint i = 0; i < 16; i += 4) {
        sum += tofloat(x[i]) + tofloat(x[i+1]) + tofloat(x[i+2]) + tofloat(x[i+3]);
        v[i]   = tofloat(x[i]);
        v[i+1] = tofloat(x[i+1]) / 16.0f;
        v[i+2] = tofloat(x[i+2]) / 256.0f;
        v[i+3] = tofloat(x[i+3]) / 4096.0f;
    }
    return sum;
}

__device__ inline float dot16(const uint8_t* w, float* x, float s, float b, float sum) {
    const uint16_t* p = (const uint16_t*)w;
    float a = 0.0f;
    for (uint j = 0; j < 4; ++j)
        a += x[4*j]   * (p[j] & 15u)
           + x[4*j+1] * (p[j] & 240u)
           + x[4*j+2] * (p[j] & 3840u)
           + x[4*j+3] * (p[j] & 61440u);
    return s * a + sum * b;
}

// One warp-group of 32 threads computes 4 rows cooperatively; the loop is
// indexed by thread id, not hardware wave, so it is wave32/wave64 agnostic.
__global__ void k_q4(Matrix m, const bf16* x, bf16* out) {
    int groups = blockDim.x / 32;   // 32-thread warp groups (wave-size agnostic)
    int lane = threadIdx.x & 31;
    int sg = threadIdx.x >> 5;
    for (int first = sg * 4; first < m.n; first += groups * 4) {
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int block = 0; block < m.k; block += 512) {
            int scalar = block + lane * 16;
            float v[16];
            float sum = load16(x + scalar, v);
            for (int r = 0; r < 4; ++r) {
                int row = first + r;
                int gi = row * (m.k / 64) + scalar / 64;
                a[r] += dot16(m.w + (size_t)row * (m.k / 2) + scalar / 2,
                              v, tofloat(m.s[gi]), tofloat(m.b[gi]), sum);
            }
        }
        for (int r = 0; r < 4; ++r) {
            float v = simd_sum(a[r]);
            if (lane == 0) out[first + r] = tobf(v);
        }
    }
}

// ---- RMSNorm (megakernel rms), 4 elements per thread ----
// hidden elements, hidden % (threads*4) == 0.
__global__ void k_rms(const bf16* x, const bf16* w, bf16* out, int hidden, float eps) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    float squared = 0.0f;
    for (int j = 0; j < 4; ++j) {
        float v = tofloat(x[tid * 4 + j]);
        squared += v * v;
    }
    float sum = simd_sum(squared);          // warp reduce
    int lane = tid & 31, sg = tid >> 5;
    int warps = blockDim.x >> 5;
    if (lane == 0) temp[sg] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < warps; ++i) total += temp[i];
        // metal::precise::rsqrt equivalent: full-precision reciprocal sqrt
        temp[32] = 1.0f / sqrtf(total / (float)hidden + eps);
    }
    __syncthreads();
    for (int j = 0; j < 4; ++j) {
        int i = tid * 4 + j;
        out[i] = tobf(tofloat(w[i]) * tofloat(x[i]) * temp[32]);
    }
}

// ---- head_norm: per-head RMSNorm + NeoX-half RoPE (megakernel head_norm) ----
// One warp handles one head; heads strided by gridDim.x. head_dim = 128,
// 4 elements per lane. rotary=true applies pair/half rotation.
__global__ void k_head_norm(bf16* x, const bf16* w, int heads, int rotary,
                            int position, float logbase, float eps) {
    int lane = threadIdx.x & 31;
    for (int h = blockIdx.x; h < heads; h += gridDim.x) {
        bf16* p = x + (size_t)h * 128;
        float squared = 0.0f;
        for (int j = 0; j < 4; ++j) { float v = tofloat(p[lane * 4 + j]); squared += v * v; }
        float inverse = 1.0f / sqrtf(simd_sum(squared) / 128.0f + eps);
        if (rotary) {
            for (int j = 0; j < 2; ++j) {
                int pair = lane * 2 + j;
                float a = tofloat(w[pair])     * tofloat(p[pair])     * inverse;
                float b = tofloat(w[pair + 64]) * tofloat(p[pair + 64]) * inverse;
                float theta = (float)position * exp2f(-(float)pair / 64.0f * logbase);
                float c = cosf(theta), s = sinf(theta);
                p[pair]     = tobf(a * c - b * s);
                p[pair + 64] = tobf(a * s + b * c);
            }
        } else {
            for (int j = 0; j < 4; ++j) {
                int d = lane * 4 + j;
                p[d] = tobf(tofloat(w[d]) * tofloat(p[d]) * inverse);
            }
        }
    }
}

// ---- attention with SINKS (megakernel attention_partial, parts=1, 1 warp) ----
// HEADS = q-heads per kv group (Flint decode: 8), head_dim 128, 32 lanes.
// Streaming softmax where the sink is the initial max/weight: literally the
// MSL `hi = sinks[first+lane]; z = 1` initialization on lanes < HEADS.
// q: [nheads*128], k/v: [nkv * count * 128], sinks: [nheads], out: [nheads*128]
constexpr int ATTN_HEADS = 8;

__global__ void k_attn_sinks(const bf16* q, const bf16* k, const bf16* v,
                             const bf16* sinks, bf16* out, int count) {
    int lane = threadIdx.x & 31;   // single warp
    const float scale = 0.08838834764831845f;  // 1/sqrt(128), literal
    for (int kv = 0; kv < 4; ++kv) {
        int first = kv * ATTN_HEADS;
        float query[ATTN_HEADS][4], acc[ATTN_HEADS][4];
        float hi = lane < ATTN_HEADS ? tofloat(sinks[first + lane]) : -3.402823466e+38f;
        float z  = lane < ATTN_HEADS ? 1.0f : 0.0f;
        for (int h = 0; h < ATTN_HEADS; ++h)
            for (int j = 0; j < 4; ++j) {
                query[h][j] = tofloat(q[(size_t)(first + h) * 128 + lane * 4 + j]) * scale;
                acc[h][j] = 0.0f;
            }
        for (int token = 0; token < count; ++token) {
            size_t at = ((size_t)kv * count + token) * 128 + lane * 4;
            float key[4], value[4];
            for (int j = 0; j < 4; ++j) { key[j] = tofloat(k[at + j]); value[j] = tofloat(v[at + j]); }
            // All lanes reduce every head score (no divergence at simd_sum —
            // the MSL original also computes all HEADS scores per lane).
            float scores[ATTN_HEADS];
            for (int h = 0; h < ATTN_HEADS; ++h) {
                float score = 0.0f;
                for (int j = 0; j < 4; ++j) score += query[h][j] * key[j];
                scores[h] = simd_sum(score);
            }
            float old = 0.0f, add = 0.0f;
            if (lane < ATTN_HEADS) {
                float score = scores[lane], next = fmaxf(hi, score);
                float low = expf(-fabsf(score - hi));  // metal::fast::exp equivalent
                bool wins = score > hi;
                old = wins ? low : 1.0f;
                add = wins ? 1.0f : low;
                z = z * old + add;
                hi = next;
            }
            for (int h = 0; h < ATTN_HEADS; ++h) {
                float factor = __shfl_sync(0xffffffffffffffffull, old, (unsigned)h);
                float weight = __shfl_sync(0xffffffffffffffffull, add, (unsigned)h);
                for (int j = 0; j < 4; ++j)
                    acc[h][j] = acc[h][j] * factor + weight * value[j];
            }
        }
        // parts == 1: merge degenerates to out = acc / z (attention_merge with one partition)
        for (int h = 0; h < ATTN_HEADS; ++h)
            for (int j = 0; j < 4; ++j) {
                int head = first + h;
                int elem = lane * 4 + j;
                float value = acc[h][j];
                float zn = __shfl_sync(0xffffffffffffffffull, z, (unsigned)h);
                out[(size_t)head * 128 + elem] = tobf(value / zn);
            }
    }
}

// ---- C ABI wrappers (host) ----
static void check(hipError_t e) {
    if (e != hipSuccess) { fprintf(stderr, "HIP error %d: %s\n", (int)e, hipGetErrorString(e)); }
}

// ---- Host-pointer wrappers: single-runtime process, no GPU framework needed ----
static int stage(size_t bytes, void** dev, const void* host) {
    if (hipMalloc(dev, bytes) != hipSuccess) return -1;
    if (host && hipMemcpy(*dev, host, bytes, hipMemcpyHostToDevice) != hipSuccess) return -2;
    return 0;
}

extern "C" int flint_q4_host(const uint16_t* h_w, const uint16_t* h_s, const uint16_t* h_b,
                             const uint16_t* h_x, uint16_t* h_out, int n, int k, int groups) {
    size_t wb = (size_t)n * (k / 2), sb = (size_t)n * (k / 64) * 2, xb = (size_t)k * 2, ob = (size_t)n * 2;
    void *dw, *ds, *db, *dx, *dout;
    if (stage(wb, &dw, h_w)) return -1;
    if (stage(sb, &ds, h_s)) return -1;
    if (stage(sb, &db, h_b)) return -1;
    if (stage(xb, &dx, h_x)) return -1;
    if (stage(ob, &dout, nullptr)) return -1;
    Matrix m{ (const uint8_t*)dw, (const bf16*)ds, (const bf16*)db, n, k };
    hipLaunchKernelGGL(k_q4, dim3(1), dim3(groups * 32), 0, 0, m, (const bf16*)dx, (bf16*)dout);
    hipDeviceSynchronize();
    hipError_t e = hipGetLastError();
    if (e == hipSuccess) e = hipMemcpy(h_out, dout, ob, hipMemcpyDeviceToHost);
    hipFree(dw); hipFree(ds); hipFree(db); hipFree(dx); hipFree(dout);
    return e == hipSuccess ? 0 : (int)e;
}

extern "C" int flint_rms_host(const uint16_t* h_x, const uint16_t* h_w, uint16_t* h_out,
                              int hidden, float eps) {
    void *dx, *dw, *dout;
    size_t bytes = (size_t)hidden * 2;
    if (stage(bytes, &dx, h_x)) return -1;
    if (stage(bytes, &dw, h_w)) return -1;
    if (stage(bytes, &dout, nullptr)) return -1;
    hipLaunchKernelGGL(k_rms, dim3(1), dim3(hidden / 4), 33 * sizeof(float), 0,
                       (const bf16*)dx, (const bf16*)dw, (bf16*)dout, hidden, eps);
    hipDeviceSynchronize();
    hipError_t e = hipGetLastError();
    if (e == hipSuccess) e = hipMemcpy(h_out, dout, bytes, hipMemcpyDeviceToHost);
    hipFree(dx); hipFree(dw); hipFree(dout);
    return e == hipSuccess ? 0 : (int)e;
}

extern "C" int flint_head_norm_host(uint16_t* h_x, const uint16_t* h_w, int heads, int rotary,
                                    int position, float logbase, float eps) {
    void *dx, *dw;
    size_t bytes = (size_t)heads * 128 * 2;
    if (stage(bytes, &dx, h_x)) return -1;
    if (stage(128 * 2, &dw, h_w)) return -1;
    hipLaunchKernelGGL(k_head_norm, dim3(heads), dim3(32), 0, 0,
                       (bf16*)dx, (const bf16*)dw, heads, rotary != 0, position, logbase, eps);
    hipDeviceSynchronize();
    hipError_t e = hipGetLastError();
    if (e == hipSuccess) e = hipMemcpy(h_x, dx, bytes, hipMemcpyDeviceToHost);
    hipFree(dx); hipFree(dw);
    return e == hipSuccess ? 0 : (int)e;
}

extern "C" int flint_attn_sinks_host(const uint16_t* h_q, const uint16_t* h_k, const uint16_t* h_v,
                                     const uint16_t* h_sinks, uint16_t* h_out, int count) {
    size_t qb = (size_t)32 * 128 * 2, kvb = (size_t)4 * count * 128 * 2, sb = (size_t)32 * 2, ob = qb;
    void *dq, *dk, *dv, *dsinks, *dout;
    if (stage(qb, &dq, h_q)) return -1;
    if (stage(kvb, &dk, h_k)) return -1;
    if (stage(kvb, &dv, h_v)) return -1;
    if (stage(sb, &dsinks, h_sinks)) return -1;
    if (stage(ob, &dout, nullptr)) return -1;
    hipLaunchKernelGGL(k_attn_sinks, dim3(1), dim3(32), 0, 0,
                       (const bf16*)dq, (const bf16*)dk, (const bf16*)dv,
                       (const bf16*)dsinks, (bf16*)dout, count);
    hipDeviceSynchronize();
    hipError_t e = hipGetLastError();
    if (e == hipSuccess) e = hipMemcpy(h_out, dout, ob, hipMemcpyDeviceToHost);
    hipFree(dq); hipFree(dk); hipFree(dv); hipFree(dsinks); hipFree(dout);
    return e == hipSuccess ? 0 : (int)e;
}

extern "C" int flint_warp_size() {
    int v = 0;
    check(hipDeviceGetAttribute(&v, (hipDeviceAttribute_t)15 /* WarpSize */, 0));
    return v;
}

extern "C" int flint_q4(const void* w, const void* s, const void* b,
                        const void* x, void* out, int n, int k, int groups) {
    Matrix m{ (const uint8_t*)w, (const bf16*)s, (const bf16*)b, n, k };
    int threads = groups * 32;
    k_q4<<<1, threads>>>(m, (const bf16*)x, (bf16*)out);
    check(hipDeviceSynchronize());
    return (int)hipGetLastError();
}

extern "C" int flint_rms(const void* x, const void* w, void* out, int hidden, float eps) {
    int threads = hidden / 4;
    size_t shmem = 33 * sizeof(float);
    k_rms<<<1, threads, shmem>>>((const bf16*)x, (const bf16*)w, (bf16*)out, hidden, eps);
    check(hipDeviceSynchronize());
    return (int)hipGetLastError();
}

extern "C" int flint_head_norm(void* x, const void* w, int heads, int rotary,
                               int position, float logbase, float eps) {
    k_head_norm<<<heads, 32>>>((bf16*)x, (const bf16*)w, heads, rotary != 0,
                               position, logbase, eps);
    check(hipDeviceSynchronize());
    return (int)hipGetLastError();
}

extern "C" int flint_attn_sinks(const void* q, const void* k, const void* v,
                                const void* sinks, void* out, int count) {
    k_attn_sinks<<<1, 32>>>((const bf16*)q, (const bf16*)k, (const bf16*)v,
                            (const bf16*)sinks, (bf16*)out, count);
    check(hipDeviceSynchronize());
    return (int)hipGetLastError();
}
