// rec-21a step-2 timing probes: W4A8 gemv candidates vs the dequant-FMA baseline (gfx1150, dusk).
// K=2048 (row = 1024 B packed nibbles, low nibble = even element, 32 g64 scale/bias pairs), one warp per row,
// each lane owns one g64 group. A-side activations come pre-packed as int32 dwords of 4 s8 (emitted by the
// quantize pass / host); B-side weight nibbles are unpacked in-kernel per dot4 unit (scalar lane packing —
// the conservative unpack cost; vectorized perm tricks can only improve it).
// P0 baseline dequant-FMA | P1 v_dot4_i32_iu8 + unpack | P3 quantize pass (fp32 -> s8 + g16 scales + A-dwords).
// Host launchers follow the hip_phases_test.c ABI convention (return int rc, grid derived from n).
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

#define K 2048
#define ROWB (K / 2)      // 1024 packed bytes per row
#define NG64 (K / 64)     // 32
#define NG16 (K / 16)     // 128 activation groups
#define NA4 (K / 4)       // 512 packed A dwords per vector

__device__ __forceinline__ float bf(uint16_t h) {
    return __builtin_bit_cast(float, (uint32_t)h << 16);
}
__device__ __forceinline__ int dot4(int a, int b, int c) {
    int d;  // V_DOT4_I32_I8: A = 4x signed i8 (activations -127..127), B = 4x signed i8 (weight nibbles 0..15,
    asm("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));  // identical under signed interp)
    return d;
}

__global__ void k_p0_dequant_fma(const uint8_t* __restrict__ w, const uint16_t* __restrict__ s,
                                 const uint16_t* __restrict__ b, const float* __restrict__ x,
                                 float* __restrict__ y, int n) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n) return;
    const uint8_t* wr = w + (size_t)row * ROWB;
    const int g = lane;
    float d = 0.0f, sx = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const uint8_t byte = wr[g * 32 + j];
        const float w0 = (float)(byte & 0xF), w1 = (float)(byte >> 4);
        const float x0 = x[g * 64 + 2 * j], x1 = x[g * 64 + 2 * j + 1];
        d += w0 * x0 + w1 * x1;  sx += x0 + x1;
    }
    const float sv = bf(s[(size_t)row * NG64 + g]);
    const float bv = bf(b[(size_t)row * NG64 + g]);
    float acc = fmaf(sv, d, sx * bv);
    for (int off = 16; off; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) y[row] = acc;
}

__global__ void k_p1_dot4_unpack(const uint8_t* __restrict__ w, const uint16_t* __restrict__ s,
                                 const uint16_t* __restrict__ b, const int* __restrict__ qa4,
                                 const float* __restrict__ sa, float* __restrict__ y, int n) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n) return;
    const uint8_t* wr = w + (size_t)row * ROWB;
    const int g = lane;
    const float sw = bf(s[(size_t)row * NG64 + g]);
    const float bw = bf(b[(size_t)row * NG64 + g]);
    float acc = 0.0f;
    #pragma unroll
    for (int sg16 = 0; sg16 < 4; ++sg16) {         // 4 activation g16 groups inside this g64 group
        int D = 0, A = 0;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {              // 4 dot4 ops of 4 elements each = one g16 group
            const int e0 = sg16 * 16 + u * 4;
            const int Apack = qa4[(g * 64 + e0) >> 2];  // GLOBAL dword index: lane g covers elements g*64..g*64+63
            int Bpack = 0;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int elem = e0 + t;
                const uint8_t byte = wr[g * 32 + elem / 2];
                const uint8_t wv = (elem & 1) ? (byte >> 4) : (byte & 0xF);
                Bpack |= int(wv) << (8 * t);
            }
            D = dot4(Apack, Bpack, D);
            A = dot4(Apack, 0x01010101, A);
        }
        const float sav = sa[g * 4 + sg16];
        acc = fmaf(sw * sav, (float)D, fmaf(bw * sav, (float)A, acc));
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) y[row] = acc;
}

__global__ void k_p3_quantize(const float* __restrict__ x, int* __restrict__ qa4,
                              float* __restrict__ sa, int nvec) {
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (v >= nvec) return;
    const float* xv = x + (size_t)v * K;
    for (int sg16 = lane; sg16 < NG16; sg16 += 32) {
        float m = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) m = fmaxf(m, fabsf(xv[sg16 * 16 + j]));
        const float sc = fmaxf(m, 1e-30f) / 127.0f;
        int pack = 0;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int q = (int)fminf(fmaxf(rintf(xv[sg16 * 16 + j] / sc), -127.0f), 127.0f);
            if ((j & 3) == 0) pack = q & 0xFF; else pack |= (q & 0xFF) << (8 * (j & 3));
            if ((j & 3) == 3) qa4[(size_t)v * NA4 + sg16 * 4 + (j >> 2)] = pack;
        }
        sa[(size_t)v * NG16 + sg16] = sc;
    }
}

static constexpr int BLOCK = 256;   // 8 warps = 8 rows per block

extern "C" int p0_dequant_fma(const void* w, const void* s, const void* b, const void* x, void* y, int n) {
    const int grid = (n * 32 + BLOCK - 1) / BLOCK;
    k_p0_dequant_fma<<<grid, BLOCK>>>(static_cast<const uint8_t*>(w), static_cast<const uint16_t*>(s),
                                      static_cast<const uint16_t*>(b), static_cast<const float*>(x),
                                      static_cast<float*>(y), n);
    return (int)hipGetLastError();
}

extern "C" int p1_dot4_unpack(const void* w, const void* s, const void* b, const void* qa4, const void* sa,
                              void* y, int n) {
    const int grid = (n * 32 + BLOCK - 1) / BLOCK;
    k_p1_dot4_unpack<<<grid, BLOCK>>>(static_cast<const uint8_t*>(w), static_cast<const uint16_t*>(s),
                                      static_cast<const uint16_t*>(b), static_cast<const int*>(qa4),
                                      static_cast<const float*>(sa), static_cast<float*>(y), n);
    return (int)hipGetLastError();
}

extern "C" int p3_quantize(const void* x, void* qa4, void* sa, int nvec) {
    const int grid = (nvec * 32 + BLOCK - 1) / BLOCK;
    k_p3_quantize<<<grid, BLOCK>>>(static_cast<const float*>(x), static_cast<int*>(qa4),
                                   static_cast<float*>(sa), nvec);
    return (int)hipGetLastError();
}
