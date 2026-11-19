// Flint Mid-8L & General Multi-Workgroup Persistent Decode Megakernel + Fused INT4 Matmul
// (rec-tan Phase 2b / rec-aip prep).
//
// Consumes the exact zero-copy layout contract in docs/rocm-model-layer.md s2 & s4:
//   - INT4 affine weights: U8 [..., out, K/2] (low nibble = even element),
//     scales/biases BF16 [..., out, K/64] (group_size = 64).
//     Expert strides: packed = e * out * (K/2) bytes; s/b = e * out * (K/64) ELEMENTS of bf16*.
//     MSL qdot pre-scale period 4: x[4*u + nib] / 16^nib, bias fold via sum(x_group) * bias.
//   - INT8 affine embedding: U8 [V, H] (1 byte/elem), scales/biases BF16 [V, H/64].
//   - Dense params: norms, q_norm, k_norm, softmax_offset (sinks), router_w [E, H] (BF16),
//     expert_bias [E] (BF16).
//   - KV cache: per-layer head-major BF16:
//     * sliding layers (even): [2 (K|V)][kv_heads][513][head_dim], ring slot = pos % 513
//     * full layers (odd):     [2 (K|V)][kv_heads][cap][head_dim], linear slot = pos
//
// Also exports standalone fused INT4 / INT8 dequant-matmul device & host entrypoints
// (`flint_q4_gemv_device`, `flint_q4_gemv_host`) for rec-aip.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

typedef __hip_bfloat16 bf16;

__device__ inline float tofloat(bf16 v) { return __bfloat162float(v); }
__device__ inline bf16 tobf(float v) { return __float2bfloat16(v); }
__device__ inline float silu_f(float v) { return v / (1.0f + expf(-v)); }

// 32-lane logical warp reductions (works on both RDNA wave32 and CDNA wave64
// because __shfl_xor_sync with offsets 16..1 stays within aligned 32-lane halves).
__device__ inline float simd_sum(float v) {
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        v += __shfl_xor_sync(0xffffffffffffffffull, v, off);
    return v;
}

__device__ inline float simd_max(float v) {
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffffffffffull, v, off));
    return v;
}

// Device-wide spin barrier across `workers` resident workgroups (MSL sync_all).
__device__ inline bool sync_all(uint32_t* control, int tid, uint32_t& phase,
                                uint32_t* ready, int workers) {
    __threadfence();
    __syncthreads();
    if (workers <= 1) {
        ++phase;
        return true;
    }
    if (tid == 0) {
        uint32_t arrived = __hip_atomic_fetch_add(control, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (arrived == (uint32_t)(workers - 1)) {
            __hip_atomic_store(control, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __threadfence();
            __hip_atomic_store(control + 1, phase + 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        uint32_t spins = 0;
        while (__hip_atomic_load(control + 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == phase &&
               __hip_atomic_load(control + 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 0u &&
               spins < 50000000u) {
            ++spins;
        }
        uint32_t ok = (__hip_atomic_load(control + 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) > phase) ? 1u : 0u;
        ready[0] = ok;
        if (!ok) __hip_atomic_store(control + 2, phase + 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    __threadfence();
    ++phase;
    return ready[0] != 0u;
}

// ---- MSL load16 / dot16 / q4 building blocks (period-4 pre-scale) ----
__device__ inline float load16_f(const float* x, float* v) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i += 4) {
        float x0 = x[i], x1 = x[i + 1], x2 = x[i + 2], x3 = x[i + 3];
        sum += x0 + x1 + x2 + x3;
        v[i]     = x0;
        v[i + 1] = x1 * (1.0f / 16.0f);
        v[i + 2] = x2 * (1.0f / 256.0f);
        v[i + 3] = x3 * (1.0f / 4096.0f);
    }
    return sum;
}

__device__ inline float load16_bf(const bf16* x, float* v) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i += 4) {
        float x0 = tofloat(x[i]), x1 = tofloat(x[i + 1]), x2 = tofloat(x[i + 2]), x3 = tofloat(x[i + 3]);
        sum += x0 + x1 + x2 + x3;
        v[i]     = x0;
        v[i + 1] = x1 * (1.0f / 16.0f);
        v[i + 2] = x2 * (1.0f / 256.0f);
        v[i + 3] = x3 * (1.0f / 4096.0f);
    }
    return sum;
}

__device__ inline float dot16(const uint8_t* w, const float* x, float s, float b, float sum) {
    const uint16_t* p = (const uint16_t*)w;
    float a = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint16_t u = p[j];
        a += x[4 * j]     * (float)(u & 15u)
           + x[4 * j + 1] * (float)(u & 240u)
           + x[4 * j + 2] * (float)(u & 3840u)
           + x[4 * j + 3] * (float)(u & 61440u);
    }
    return s * a + sum * b;
}

// Multi-workgroup q4 matmul (K is a multiple of 512, e.g. 512, 1024, 2048, 4096).
// Each 32-thread warp `sg` (0..groups-1) computes 4 rows at a time across K in 512-wide blocks.
// Writes bf16 rounded results to `out_bf` and/or float (unrounded or bf16-rounded) to `out_f`.
template <int K, bool ROUND_BF16 = false>
__device__ inline void q4_block512(const uint8_t* w, const bf16* s, const bf16* b,
                                   const float* x, bf16* out_bf, float* out_f,
                                   int n, int lane, int sg, int groups) {
    static_assert(K % 512 == 0, "q4_block512 requires K % 512 == 0");
    for (int first = sg * 4; first < n; first += groups * 4) {
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int block = 0; block < K; block += 512) {
            int scalar = block + lane * 16;
            float v[16];
            float sum = load16_f(x + scalar, v);
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int row = first + r;
                if (row < n) {
                    int gi = row * (K / 64) + scalar / 64;
                    a[r] += dot16(w + (size_t)row * (K / 2) + scalar / 2,
                                  v, tofloat(s[gi]), tofloat(b[gi]), sum);
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            float val = simd_sum(a[r]);
            if (lane == 0 && first + r < n) {
                if (out_bf) out_bf[first + r] = tobf(val);
                if (out_f)  out_f[first + r]  = ROUND_BF16 ? tofloat(tobf(val)) : val;
            }
        }
    }
}

// ---- Mid-8L Flint constants ----
constexpr int MID_NL = 8;
constexpr int MID_HID = 2048;
constexpr int MID_HEADS = 32;
constexpr int MID_NKV = 4;
constexpr int MID_HDIM = 128;
constexpr int MID_QDIM = MID_HEADS * MID_HDIM;   // 4096
constexpr int MID_KVDIM = MID_NKV * MID_HDIM;    // 512
constexpr int MID_DENSE_I = 2048;
constexpr int MID_MOE_I = 512;
constexpr int MID_SH_I = 1024;
constexpr int MID_NEXP = 32;
constexpr int MID_TOPK = 2;
constexpr int MID_VOCAB = 150144;
constexpr int MID_WIN = 513;
constexpr float MID_EPS = 1e-6f;
constexpr float MID_ATTN_SCALE = 0.08838834764831845f; // 1/sqrt(128)
constexpr float MID_ROPE_LOGBASE = 13.28771237954945f; // log2(10000)

// ---- Mid-8L bank offset table indices ----
// Each Q4 tensor has 1 offset entry pointing to packed [out, K/2];
// scales immediately follow at + out*(K/2), biases at + out*(K/2) + out*(K/64)*2.
// Embed (INT8): packed [V, H] (V*H bytes), scales [V, H/64] bf16, biases [V, H/64] bf16.
// Per layer l (0..7):
//   Dense bf16:
//     l*7 + 0: input_layernorm [2048] bf16
//     l*7 + 1: post_attention_layernorm [2048] bf16
//     l*7 + 2: q_norm [128] bf16
//     l*7 + 3: k_norm [128] bf16
//     l*7 + 4: sinks (softmax_offset) [32] bf16
//     l*7 + 5: router_w [32, 2048] bf16 (0 for L0)
//     l*7 + 6: router_b [32] bf16 (0 for L0)
//   Global dense bf16:
//     56: final_norm [2048] bf16
//   Embed INT8:
//     57: embed_tokens (INT8 packed + bf16 s + bf16 b)
//   Per layer l (0..7) Q4 tensors (10 slots per layer, base = 58 + l*10):
//     +0: q_proj    [4096, 2048]
//     +1: k_proj    [512, 2048]
//     +2: v_proj    [512, 2048]
//     +3: o_proj    [2048, 4096]
//     +4: gate_proj [2048, 2048] (L0) or sw_gate [32*512, 2048] (L1..7)
//     +5: up_proj   [2048, 2048] (L0) or sw_up   [32*512, 2048] (L1..7)
//     +6: down_proj [2048, 2048] (L0) or sw_down [32*2048, 512] (L1..7)
//     +7: sh_gate   [1024, 2048] (L1..7)
//     +8: sh_up     [1024, 2048] (L1..7)
//     +9: sh_down   [2048, 1024] (L1..7)
//   Global Q4:
//     138: lm_head  [150144, 2048]
//   Total offset entries = 139 (off[139] = total bank size in bytes).
constexpr int MID_OFF_FINAL_NORM = 56;
constexpr int MID_OFF_EMBED = 57;
constexpr int MID_OFF_LAYER_Q4_BASE = 58;
constexpr int MID_OFF_LM_HEAD = 138;
constexpr int MID_NUM_OFFSETS = 140;

// 256-thread block RMSNorm over HID=2048 (8 elems/thread) from float `in_f` to shared float `out_sh`.
__device__ inline void mid_rmsnorm_2048(const float* in_f, const bf16* w_bf,
                                        float* out_sh, float* red_sh) {
    int tid = threadIdx.x; // 0..255
    int lane = tid & 31;
    int warp = tid >> 5;   // 0..7
    float sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int i = tid * 8 + j;
        float v = in_f[i];
        sq += v * v;
    }
    float wsum = simd_sum(sq);
    if (lane == 0) red_sh[warp] = wsum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) total += red_sh[w];
        red_sh[8] = 1.0f / sqrtf(total / (float)MID_HID + MID_EPS);
    }
    __syncthreads();
    float inv = red_sh[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int i = tid * 8 + j;
        out_sh[i] = in_f[i] * inv * tofloat(w_bf[i]);
    }
    __syncthreads();
}

// Per-head RMSNorm (head_dim=128) + optional NeoX-half RoPE across global warps in fp32.
// One 32-lane warp processes one head (4 elements per lane).
__device__ inline void mid_head_norm_rope(float* qk, const bf16* nw, int heads,
                                          int rotary, int position,
                                          int lane, int sg, int groups) {
    for (int h = sg; h < heads; h += groups) {
        float* p = qk + (size_t)h * MID_HDIM;
        float sq = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float v = p[lane * 4 + j];
            sq += v * v;
        }
        float inv = 1.0f / sqrtf(simd_sum(sq) / (float)MID_HDIM + MID_EPS);
        if (rotary) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int pair = lane * 2 + j;
                float a = p[pair] * inv * tofloat(nw[pair]);
                float b = p[pair + 64] * inv * tofloat(nw[pair + 64]);
                float theta = (float)position * exp2f(-(float)pair / 64.0f * MID_ROPE_LOGBASE);
                float c = cosf(theta), s = sinf(theta);
                p[pair]      = a * c - b * s;
                p[pair + 64] = a * s + b * c;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int d = lane * 4 + j;
                p[d] = p[d] * inv * tofloat(nw[d]);
            }
        }
    }
}

// Workspace layout in device memory for mid_generate:
//   bf16 buffers:
//     ws_x[2048]          : residual stream [HID]
//     ws_q[4096]          : q_proj output [QDIM]
//     ws_k[512]           : k_proj output [KVDIM]
//     ws_v[512]           : v_proj output [KVDIM]
//     ws_attn[4096]       : attention output [QDIM]
//     ws_o[2048]          : o_proj output [HID]
//     ws_gate[2048]       : dense/routed/shared gate_proj output (max 2048)
//     ws_up[2048]         : dense/routed/shared up_proj output (max 2048)
//     ws_down[3 * 2048]   : [0..2047] routed exp0 / dense down, [2048..4095] routed exp1, [4096..6143] shared down
//   float buffers:
//     ws_f_act[2048]      : SiLU(gate)*up intermediate in float (rounded to bf16)
//     ws_f_scores[32]     : router sigmoid scores [NEXP]
//     ws_f_wsel[2]        : normalized top-2 weights
//     ws_f_logits[150144] : fp32 output logits
//     ws_f_maxval[64]     : per-workgroup argmax max value
//   int buffers:
//     ws_i_sel[2]         : selected top-2 expert indices
//     ws_i_maxidx[64]     : per-workgroup argmax best index

template <typename KVT, bool PURE_FP32>
__device__ inline float kv_load(KVT v);

template <>
__device__ inline float kv_load<bf16, false>(bf16 v) { return tofloat(v); }
template <>
__device__ inline float kv_load<bf16, true>(bf16 v) { return tofloat(v); }
template <>
__device__ inline float kv_load<float, false>(float v) { return v; }
template <>
__device__ inline float kv_load<float, true>(float v) { return v; }

template <typename KVT>
__device__ inline KVT kv_store(float v);

template <>
__device__ inline bf16 kv_store<bf16>(float v) { return tobf(v); }
template <>
__device__ inline float kv_store<float>(float v) { return v; }

template <typename KVT, bool PURE_FP32>
__device__ inline void mid_generate_impl(
    const uint8_t* bank, const int64_t* off,
    KVT* kv_sliding,                         // [4][2 (K|V)][4][513][128] (layers 0,2,4,6)
    KVT* kv_full,                            // [4][2 (K|V)][4][full_cap][128] (layers 1,3,5,7)
    int full_cap,
    const int* prompt, int prompt_len,
    int* out_tokens, int n_out,
    float* out_step_logits, float* out_last_logits,
    bf16* ws_bf, float* ws_f, int* ws_i,
    uint32_t* control, int workers,
    int* out_chosen_experts = nullptr) {

    const int tid = threadIdx.x;          // 0..255
    const int lane = tid & 31;            // 0..31
    const int group = blockIdx.x;         // 0..workers-1
    const int sg = group * 8 + (tid >> 5);// 0..workers*8-1
    const int groups = workers * 8;
    const int gtid = group * 256 + tid;
    const int gthreads = workers * 256;
    const int total = prompt_len + n_out;

    // Partition device workspace pointers (float precision matching tiny_megakernel.hip)
    float* ws_x      = ws_f + 0;
    float* ws_q      = ws_f + 2048;
    float* ws_k      = ws_f + 2048 + 4096;
    float* ws_v      = ws_f + 2048 + 4096 + 512;
    float* ws_attn   = ws_f + 2048 + 4096 + 1024;
    float* ws_o      = ws_f + 2048 + 8192 + 1024;
    float* ws_gate   = ws_f + 4096 + 8192 + 1024;
    float* ws_up     = ws_f + 6144 + 8192 + 1024;
    float* ws_down   = ws_f + 8192 + 8192 + 1024; // 3 * 2048 = 6144 elems
    float* ws_f_act    = ws_f + 14336 + 8192 + 1024;
    float* ws_f_scores = ws_f_act + 2048;
    float* ws_f_wsel   = ws_f_scores + 32;
    float* ws_f_maxval = ws_f_scores + 64;
    float* ws_f_logits = ws_f_scores + 128;
    (void)ws_bf;

    int* ws_i_sel    = ws_i + 0;
    int* ws_i_maxidx = ws_i + 16;

    __shared__ float sh_xn[MID_HID];
    __shared__ float sh_red[16];
    __shared__ uint32_t sh_ready[1];
    uint32_t phase = 0;

    #define SYNC_ALL() if (!sync_all(control, tid, phase, sh_ready, workers)) return

    auto B = [&](int idx) -> const bf16* { return (const bf16*)(bank + off[idx]); };
    auto Q_W = [&](int idx) -> const uint8_t* { return bank + off[idx]; };
    auto Q_S = [&](int idx, int n, int k) -> const bf16* {
        return (const bf16*)(bank + off[idx] + (size_t)n * (k / 2));
    };
    auto Q_B = [&](int idx, int n, int k) -> const bf16* {
        return (const bf16*)(bank + off[idx] + (size_t)n * (k / 2) + (size_t)n * (k / 64) * 2);
    };

    const uint8_t* emb_w = bank + off[MID_OFF_EMBED];
    const bf16* emb_s = (const bf16*)(emb_w + (size_t)MID_VOCAB * MID_HID);
    const bf16* emb_b = (const bf16*)((const uint8_t*)emb_s + (size_t)MID_VOCAB * (MID_HID / 64) * 2);

    for (int pos = 0; pos < total; ++pos) {
        int token = (pos < prompt_len) ? prompt[pos] : out_tokens[pos - prompt_len];

        // ---- 1. INT8 affine embedding lookup -> ws_x[0..2047] ----
        for (int i = gtid; i < MID_HID; i += gthreads) {
            uint8_t q = emb_w[(size_t)token * MID_HID + i];
            size_t gi = (size_t)token * (MID_HID / 64) + (i / 64);
            ws_x[i] = (float)q * tofloat(emb_s[gi]) + tofloat(emb_b[gi]);
        }
        SYNC_ALL();

        for (int layer = 0; layer < MID_NL; ++layer) {
            const bf16* ln_w   = B(layer * 7 + 0);
            const bf16* post_w = B(layer * 7 + 1);
            const bf16* qn_w   = B(layer * 7 + 2);
            const bf16* kn_w   = B(layer * 7 + 3);
            const bf16* sinks  = B(layer * 7 + 4);
            const bf16* rw_bf  = B(layer * 7 + 5);
            const bf16* rb_bf  = B(layer * 7 + 6);

            int qbase = MID_OFF_LAYER_Q4_BASE + layer * 10;
            int is_sliding = (layer % 2 == 0);
            int rotary = is_sliding; // in Flint mid-8L, no_rope_layer_indices = odd layers

            // ---- 2. Input RMSNorm (replicated in each workgroup's sh_xn) ----
            mid_rmsnorm_2048(ws_x, ln_w, sh_xn, sh_red);

            // ---- 3. Q, K, V projections (K=2048) ----
            q4_block512<MID_HID>(Q_W(qbase + 0), Q_S(qbase + 0, MID_QDIM, MID_HID), Q_B(qbase + 0, MID_QDIM, MID_HID),
                                 sh_xn, nullptr, ws_q, MID_QDIM, lane, sg, groups);
            q4_block512<MID_HID>(Q_W(qbase + 1), Q_S(qbase + 1, MID_KVDIM, MID_HID), Q_B(qbase + 1, MID_KVDIM, MID_HID),
                                 sh_xn, nullptr, ws_k, MID_KVDIM, lane, sg, groups);
            q4_block512<MID_HID>(Q_W(qbase + 2), Q_S(qbase + 2, MID_KVDIM, MID_HID), Q_B(qbase + 2, MID_KVDIM, MID_HID),
                                 sh_xn, nullptr, ws_v, MID_KVDIM, lane, sg, groups);
            SYNC_ALL();

            // ---- 4. Per-head RMSNorm + NeoX RoPE on ws_q (32 heads) and ws_k (4 heads) ----
            mid_head_norm_rope(ws_q, qn_w, MID_HEADS, rotary, pos, lane, sg, groups);
            mid_head_norm_rope(ws_k, kn_w, MID_NKV, rotary, pos, lane, sg, groups);
            SYNC_ALL();

            // ---- 5. Write K, V to head-major KV cache (KVT = bf16 or float) ----
            int l_idx = layer / 2;
            size_t l_cap = is_sliding ? (size_t)MID_WIN : (size_t)full_cap;
            size_t slot  = is_sliding ? (size_t)(pos % MID_WIN) : (size_t)pos;
            KVT* l_kv    = is_sliding
                ? (kv_sliding + (size_t)l_idx * 2 * MID_NKV * MID_WIN * MID_HDIM)
                : (kv_full    + (size_t)l_idx * 2 * MID_NKV * (size_t)full_cap * MID_HDIM);

            for (int i = gtid; i < MID_KVDIM; i += gthreads) {
                int h = i / MID_HDIM;
                int d = i % MID_HDIM;
                size_t k_off = ((0 * MID_NKV + (size_t)h) * l_cap + slot) * MID_HDIM + d;
                size_t v_off = ((1 * MID_NKV + (size_t)h) * l_cap + slot) * MID_HDIM + d;
                l_kv[k_off] = kv_store<KVT>(ws_k[i]);
                l_kv[v_off] = kv_store<KVT>(ws_v[i]);
            }
            SYNC_ALL();

            // ---- 6. Attention with Sinks (32 warps across workgroups, 1 q-head per warp) ----
            int t_start = is_sliding ? ((pos - (MID_WIN - 1) > 0) ? (pos - (MID_WIN - 1)) : 0) : 0;
            for (int h = sg; h < MID_HEADS; h += groups) {
                int kvh = h / (MID_HEADS / MID_NKV); // h / 8
                const KVT* kc = l_kv + (0 * MID_NKV + (size_t)kvh) * l_cap * MID_HDIM;
                const KVT* vc = l_kv + (1 * MID_NKV + (size_t)kvh) * l_cap * MID_HDIM;
                float qreg[4], acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float qv = ws_q[(size_t)h * MID_HDIM + lane * 4 + j];
                    if constexpr (!PURE_FP32) qv = tofloat(tobf(qv));
                    qreg[j] = qv * MID_ATTN_SCALE;
                }
                float hi = tofloat(sinks[h]);
                float z = 1.0f;
                for (int t = t_start; t <= pos; ++t) {
                    size_t t_slot = is_sliding ? (size_t)(t % MID_WIN) : (size_t)t;
                    const KVT* kt = kc + t_slot * MID_HDIM + lane * 4;
                    const KVT* vt = vc + t_slot * MID_HDIM + lane * 4;
                    float partial = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) partial += qreg[j] * kv_load<KVT, PURE_FP32>(kt[j]);
                    float score = simd_sum(partial);
                    float next = fmaxf(hi, score);
                    float low = expf(-fabsf(score - hi));
                    bool wins = score > hi;
                    float old_f = wins ? low : 1.0f;
                    float add_f = wins ? 1.0f : low;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        acc[j] = acc[j] * old_f + add_f * kv_load<KVT, PURE_FP32>(vt[j]);
                    z = z * old_f + add_f;
                    hi = next;
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    ws_attn[(size_t)h * MID_HDIM + lane * 4 + j] = acc[j] / z;
            }
            SYNC_ALL();

            // ---- 7. O projection (K=4096 -> N=2048) + residual ----
            q4_block512<MID_QDIM>(Q_W(qbase + 3), Q_S(qbase + 3, MID_HID, MID_QDIM), Q_B(qbase + 3, MID_HID, MID_QDIM),
                                  ws_attn, nullptr, ws_o, MID_HID, lane, sg, groups);
            SYNC_ALL();
            for (int i = gtid; i < MID_HID; i += gthreads)
                ws_x[i] += ws_o[i];
            SYNC_ALL();

            // ---- 8. Post-attention RMSNorm -> sh_xn ----
            mid_rmsnorm_2048(ws_x, post_w, sh_xn, sh_red);

            // ---- 9. MLP (Layer 0 Dense vs Layers 1..7 MoE) ----
            if (layer == 0) {
                if (group == 0 && tid == 0 && out_chosen_experts != nullptr) {
                    int* exp_dst = out_chosen_experts + ((size_t)pos * MID_NL + layer) * MID_TOPK;
                    exp_dst[0] = -1;
                    exp_dst[1] = -1;
                }
                q4_block512<MID_HID>(Q_W(qbase + 4), Q_S(qbase + 4, MID_DENSE_I, MID_HID), Q_B(qbase + 4, MID_DENSE_I, MID_HID),
                                     sh_xn, nullptr, ws_gate, MID_DENSE_I, lane, sg, groups);
                q4_block512<MID_HID>(Q_W(qbase + 5), Q_S(qbase + 5, MID_DENSE_I, MID_HID), Q_B(qbase + 5, MID_DENSE_I, MID_HID),
                                     sh_xn, nullptr, ws_up, MID_DENSE_I, lane, sg, groups);
                SYNC_ALL();
                for (int i = gtid; i < MID_DENSE_I; i += gthreads) {
                    if constexpr (PURE_FP32) {
                        ws_f_act[i] = silu_f(ws_gate[i]) * ws_up[i];
                    } else {
                        bf16 act = tobf(silu_f(tofloat(tobf(ws_gate[i]))) * tofloat(tobf(ws_up[i])));
                        ws_f_act[i] = tofloat(act);
                    }
                }
                SYNC_ALL();
                q4_block512<MID_DENSE_I>(Q_W(qbase + 6), Q_S(qbase + 6, MID_HID, MID_DENSE_I), Q_B(qbase + 6, MID_HID, MID_DENSE_I),
                                         ws_f_act, nullptr, ws_down, MID_HID, lane, sg, groups);
                SYNC_ALL();
                for (int i = gtid; i < MID_HID; i += gthreads)
                    ws_x[i] += ws_down[i];
                SYNC_ALL();
            } else {
                // MoE router: 32 experts, 1 warp per expert (sg < 32)
                for (int e = sg; e < MID_NEXP; e += groups) {
                    const bf16* rrow = rw_bf + (size_t)e * MID_HID;
                    float partial = 0.0f;
                    for (int i = lane * 4; i < MID_HID; i += 128) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j)
                            partial += tofloat(rrow[i + j]) * sh_xn[i + j];
                    }
                    float dot = simd_sum(partial);
                    if (lane == 0) {
                        float d_in = PURE_FP32 ? dot : tofloat(tobf(dot));
                        ws_f_scores[e] = 1.0f / (1.0f + expf(-d_in));
                    }
                }
                SYNC_ALL();
                if (group == 0 && tid == 0) {
                    int top0 = 0, top1 = 0;
                    float tv0 = -1e30f, tv1 = -1e30f;
                    for (int e = 0; e < MID_NEXP; ++e) {
                        float choice = ws_f_scores[e] + tofloat(rb_bf[e]);
                        if (choice > tv0) {
                            tv1 = tv0; top1 = top0;
                            tv0 = choice; top0 = e;
                        } else if (choice > tv1) {
                            tv1 = choice; top1 = e;
                        }
                    }
                    ws_i_sel[0] = top0;
                    ws_i_sel[1] = top1;
                    if (out_chosen_experts != nullptr) {
                        int* exp_dst = out_chosen_experts + ((size_t)pos * MID_NL + layer) * MID_TOPK;
                        exp_dst[0] = top0;
                        exp_dst[1] = top1;
                    }
                    float s0 = ws_f_scores[top0], s1 = ws_f_scores[top1];
                    float wsum = s0 + s1 + 1e-20f;
                    ws_f_wsel[0] = s0 / wsum;
                    ws_f_wsel[1] = s1 / wsum;
                }
                SYNC_ALL();

                // Routed top-2 experts (MOE_I = 512, HID = 2048)
                const uint8_t* sw_gw = Q_W(qbase + 4);
                const bf16*    sw_gs = Q_S(qbase + 4, MID_NEXP * MID_MOE_I, MID_HID);
                const bf16*    sw_gb = Q_B(qbase + 4, MID_NEXP * MID_MOE_I, MID_HID);
                const uint8_t* sw_uw = Q_W(qbase + 5);
                const bf16*    sw_us = Q_S(qbase + 5, MID_NEXP * MID_MOE_I, MID_HID);
                const bf16*    sw_ub = Q_B(qbase + 5, MID_NEXP * MID_MOE_I, MID_HID);
                const uint8_t* sw_dw = Q_W(qbase + 6);
                const bf16*    sw_ds = Q_S(qbase + 6, MID_NEXP * MID_HID, MID_MOE_I);
                const bf16*    sw_db = Q_B(qbase + 6, MID_NEXP * MID_HID, MID_MOE_I);

                for (int t = 0; t < MID_TOPK; ++t) {
                    int e = ws_i_sel[t];
                    // Expert strides: bytes for packed uint8*, ELEMENTS for bf16* scales/biases
                    size_t gw_off = (size_t)e * MID_MOE_I * (MID_HID / 2);
                    size_t gs_off = (size_t)e * MID_MOE_I * (MID_HID / 64);
                    q4_block512<MID_HID>(sw_gw + gw_off, sw_gs + gs_off, sw_gb + gs_off,
                                         sh_xn, nullptr, ws_gate, MID_MOE_I, lane, sg, groups);
                    q4_block512<MID_HID>(sw_uw + gw_off, sw_us + gs_off, sw_ub + gs_off,
                                         sh_xn, nullptr, ws_up, MID_MOE_I, lane, sg, groups);
                    SYNC_ALL();
                    for (int i = gtid; i < MID_MOE_I; i += gthreads) {
                        if constexpr (PURE_FP32) {
                            ws_f_act[i] = silu_f(ws_gate[i]) * ws_up[i];
                        } else {
                            bf16 act = tobf(silu_f(tofloat(tobf(ws_gate[i]))) * tofloat(tobf(ws_up[i])));
                            ws_f_act[i] = tofloat(act);
                        }
                    }
                    SYNC_ALL();
                    size_t dw_off = (size_t)e * MID_HID * (MID_MOE_I / 2);
                    size_t ds_off = (size_t)e * MID_HID * (MID_MOE_I / 64);
                    q4_block512<MID_MOE_I>(sw_dw + dw_off, sw_ds + ds_off, sw_db + ds_off,
                                           ws_f_act, nullptr, ws_down + t * MID_HID, MID_HID, lane, sg, groups);
                    SYNC_ALL();
                }

                // Shared expert (SH_I = 1024, HID = 2048)
                q4_block512<MID_HID>(Q_W(qbase + 7), Q_S(qbase + 7, MID_SH_I, MID_HID), Q_B(qbase + 7, MID_SH_I, MID_HID),
                                     sh_xn, nullptr, ws_gate, MID_SH_I, lane, sg, groups);
                q4_block512<MID_HID>(Q_W(qbase + 8), Q_S(qbase + 8, MID_SH_I, MID_HID), Q_B(qbase + 8, MID_SH_I, MID_HID),
                                     sh_xn, nullptr, ws_up, MID_SH_I, lane, sg, groups);
                SYNC_ALL();
                for (int i = gtid; i < MID_SH_I; i += gthreads) {
                    if constexpr (PURE_FP32) {
                        ws_f_act[i] = silu_f(ws_gate[i]) * ws_up[i];
                    } else {
                        bf16 act = tobf(silu_f(tofloat(tobf(ws_gate[i]))) * tofloat(tobf(ws_up[i])));
                        ws_f_act[i] = tofloat(act);
                    }
                }
                SYNC_ALL();
                q4_block512<MID_SH_I>(Q_W(qbase + 9), Q_S(qbase + 9, MID_HID, MID_SH_I), Q_B(qbase + 9, MID_HID, MID_SH_I),
                                      ws_f_act, nullptr, ws_down + 2 * MID_HID, MID_HID, lane, sg, groups);
                SYNC_ALL();

                float w0 = ws_f_wsel[0], w1 = ws_f_wsel[1];
                for (int i = gtid; i < MID_HID; i += gthreads) {
                    ws_x[i] += w0 * ws_down[i] + w1 * ws_down[MID_HID + i] + ws_down[2 * MID_HID + i];
                }
                SYNC_ALL();
            }
        }

        // ---- 10. Final RMSNorm -> sh_xn ----
        mid_rmsnorm_2048(ws_x, B(MID_OFF_FINAL_NORM), sh_xn, sh_red);

        // ---- 11. LM Head Q4 Matmul (150144 x 2048) -> ws_f_logits ----
        // Always execute lm_head + argmax at every position so persistent decode32
        // differential cancellation holds identically across prefill and decode steps.
        q4_block512<MID_HID>(Q_W(MID_OFF_LM_HEAD), Q_S(MID_OFF_LM_HEAD, MID_VOCAB, MID_HID), Q_B(MID_OFF_LM_HEAD, MID_VOCAB, MID_HID),
                             sh_xn, nullptr, ws_f_logits, MID_VOCAB, lane, sg, groups);
        SYNC_ALL();

        // ---- 12. Parallel Device Argmax over 150,144 vocab ----
        float local_max = -1e30f;
        int local_idx = 0;
        for (int i = gtid; i < MID_VOCAB; i += gthreads) {
            float val = ws_f_logits[i];
            if (val > local_max || (val == local_max && i < local_idx)) {
                local_max = val;
                local_idx = i;
            }
        }
        #pragma unroll
        for (int off_lane = 16; off_lane >= 1; off_lane >>= 1) {
            float other_max = __shfl_xor_sync(0xffffffffffffffffull, local_max, off_lane);
            int   other_idx = __shfl_xor_sync(0xffffffffffffffffull, local_idx, off_lane);
            if (other_max > local_max || (other_max == local_max && other_idx < local_idx)) {
                local_max = other_max;
                local_idx = other_idx;
            }
        }
        __shared__ float sh_bmax[8];
        __shared__ int   sh_bidx[8];
        if (lane == 0) {
            sh_bmax[tid >> 5] = local_max;
            sh_bidx[tid >> 5] = local_idx;
        }
        __syncthreads();
        if (tid == 0) {
            float bmax = sh_bmax[0];
            int   bidx = sh_bidx[0];
            for (int w = 1; w < 8; ++w) {
                if (sh_bmax[w] > bmax || (sh_bmax[w] == bmax && sh_bidx[w] < bidx)) {
                    bmax = sh_bmax[w];
                    bidx = sh_bidx[w];
                }
            }
            ws_f_maxval[group] = bmax;
            ws_i_maxidx[group] = bidx;
        }
        SYNC_ALL();
        int step_slot = pos - prompt_len + 1;
        if (group == 0 && tid == 0) {
            float gmax = ws_f_maxval[0];
            int   gidx = ws_i_maxidx[0];
            for (int w = 1; w < workers; ++w) {
                if (ws_f_maxval[w] > gmax || (ws_f_maxval[w] == gmax && ws_i_maxidx[w] < gidx)) {
                    gmax = ws_f_maxval[w];
                    gidx = ws_i_maxidx[w];
                }
            }
            if (step_slot >= 0 && step_slot < n_out) out_tokens[step_slot] = gidx;
        }
        if (step_slot >= 0 && step_slot < n_out && out_step_logits) {
            float* dst = out_step_logits + (size_t)step_slot * MID_VOCAB;
            for (int i = gtid; i < MID_VOCAB; i += gthreads)
                dst[i] = ws_f_logits[i];
        }
        if (pos == total - 1 && out_last_logits) {
            for (int i = gtid; i < MID_VOCAB; i += gthreads)
                out_last_logits[i] = ws_f_logits[i];
        }
        SYNC_ALL();
    }
    #undef SYNC_ALL
}

// 1. Production BF16-KV megakernel (tiny_megakernel.hip rounding style)
extern "C" __global__ __launch_bounds__(256) void mid_generate(
    const uint8_t* bank, const int64_t* off,
    bf16* kv_sliding, bf16* kv_full, int full_cap,
    const int* prompt, int prompt_len,
    int* out_tokens, int n_out,
    float* out_step_logits, float* out_last_logits,
    bf16* ws_bf, float* ws_f, int* ws_i,
    uint32_t* control, int workers,
    int* out_chosen_experts) {
    mid_generate_impl<bf16, false>(
        bank, off, kv_sliding, kv_full, full_cap,
        prompt, prompt_len, out_tokens, n_out,
        out_step_logits, out_last_logits, ws_bf, ws_f, ws_i, control, workers,
        out_chosen_experts);
}

// 2. Pure-fp32 compute + BF16-KV megakernel
extern "C" __global__ __launch_bounds__(256) void mid_generate_fp32_bf16kv(
    const uint8_t* bank, const int64_t* off,
    bf16* kv_sliding, bf16* kv_full, int full_cap,
    const int* prompt, int prompt_len,
    int* out_tokens, int n_out,
    float* out_step_logits, float* out_last_logits,
    bf16* ws_bf, float* ws_f, int* ws_i,
    uint32_t* control, int workers,
    int* out_chosen_experts) {
    mid_generate_impl<bf16, true>(
        bank, off, kv_sliding, kv_full, full_cap,
        prompt, prompt_len, out_tokens, n_out,
        out_step_logits, out_last_logits, ws_bf, ws_f, ws_i, control, workers,
        out_chosen_experts);
}

// 3. Pure-fp32 compute + FP32-KV validation megakernel (1e-4 tier)
extern "C" __global__ __launch_bounds__(256) void mid_generate_fp32kv(
    const uint8_t* bank, const int64_t* off,
    float* kv_sliding, float* kv_full, int full_cap,
    const int* prompt, int prompt_len,
    int* out_tokens, int n_out,
    float* out_step_logits, float* out_last_logits,
    bf16* ws_bf, float* ws_f, int* ws_i,
    uint32_t* control, int workers,
    int* out_chosen_experts) {
    mid_generate_impl<float, true>(
        bank, off, kv_sliding, kv_full, full_cap,
        prompt, prompt_len, out_tokens, n_out,
        out_step_logits, out_last_logits, ws_bf, ws_f, ws_i, control, workers,
        out_chosen_experts);
}

// ---- Standalone Fused INT4 Dequant-Matmul Kernel (rec-aip) ----
// Consumes the exact zero-copy checkpoint layout from docs/rocm-model-layer.md s2:
//   w: uint8_t [N, K/2], s: bf16 [N, K/64], b: bf16 [N, K/64], x: bf16 [K] -> out: bf16 or fp32 [N]
extern "C" __global__ __launch_bounds__(256) void flint_q4_gemv_kernel(
    const uint8_t* w, const bf16* s, const bf16* b,
    const bf16* x, bf16* out_bf, float* out_f, int n, int k) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int sg = blockIdx.x * (blockDim.x >> 5) + (tid >> 5);
    int groups = gridDim.x * (blockDim.x >> 5);

    for (int first = sg * 4; first < n; first += groups * 4) {
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int block = 0; block < k; block += 512) {
            int scalar = block + lane * 16;
            float v[16];
            float sum = load16_bf(x + scalar, v);
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int row = first + r;
                if (row < n) {
                    int gi = row * (k / 64) + scalar / 64;
                    a[r] += dot16(w + (size_t)row * (k / 2) + scalar / 2,
                                  v, tofloat(s[gi]), tofloat(b[gi]), sum);
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            float val = simd_sum(a[r]);
            if (lane == 0 && first + r < n) {
                if (out_bf) out_bf[first + r] = tobf(val);
                if (out_f)  out_f[first + r]  = val;
            }
        }
    }
}

extern "C" int flint_q4_gemv_device(const void* d_w, const void* d_s, const void* d_b,
                                    const void* d_x, void* d_out_bf, float* d_out_f,
                                    int n, int k, int grid_blocks) {
    if (grid_blocks < 1) {
        int warps_needed = (n + 3) / 4;
        grid_blocks = (warps_needed + 7) / 8;
        if (grid_blocks > 256) grid_blocks = 256;
    }
    hipLaunchKernelGGL(flint_q4_gemv_kernel, dim3(grid_blocks), dim3(256), 0, 0,
                       (const uint8_t*)d_w, (const bf16*)d_s, (const bf16*)d_b,
                       (const bf16*)d_x, (bf16*)d_out_bf, d_out_f, n, k);
    return (int)hipGetLastError();
}

static void chk_hip(hipError_t e, const char* what) {
    if (e != hipSuccess) fprintf(stderr, "flint_mid %s: %s\n", what, hipGetErrorString(e));
}

extern "C" int flint_q4_gemv_host(const uint8_t* h_w, const uint16_t* h_s, const uint16_t* h_b,
                                  const uint16_t* h_x, uint16_t* h_out_bf, float* h_out_f,
                                  int n, int k, int grid_blocks, int warmup, int iters, float* out_ms) {
    size_t wb = (size_t)n * (k / 2);
    size_t sb = (size_t)n * (k / 64) * sizeof(bf16);
    size_t xb = (size_t)k * sizeof(bf16);
    size_t ob_bf = (size_t)n * sizeof(bf16);
    size_t ob_f  = (size_t)n * sizeof(float);

    uint8_t* d_w = nullptr; bf16* d_s = nullptr; bf16* d_b = nullptr; bf16* d_x = nullptr;
    bf16* d_out_bf = nullptr; float* d_out_f = nullptr;
    if (hipMalloc(&d_w, wb) != hipSuccess) return -1;
    if (hipMalloc(&d_s, sb) != hipSuccess) return -1;
    if (hipMalloc(&d_b, sb) != hipSuccess) return -1;
    if (hipMalloc(&d_x, xb) != hipSuccess) return -1;
    if (h_out_bf && hipMalloc(&d_out_bf, ob_bf) != hipSuccess) return -1;
    if (h_out_f  && hipMalloc(&d_out_f, ob_f) != hipSuccess) return -1;

    chk_hip(hipMemcpy(d_w, h_w, wb, hipMemcpyHostToDevice), "q4 H2D w");
    chk_hip(hipMemcpy(d_s, h_s, sb, hipMemcpyHostToDevice), "q4 H2D s");
    chk_hip(hipMemcpy(d_b, h_b, sb, hipMemcpyHostToDevice), "q4 H2D b");
    chk_hip(hipMemcpy(d_x, h_x, xb, hipMemcpyHostToDevice), "q4 H2D x");

    for (int i = 0; i < warmup; ++i) {
        flint_q4_gemv_device(d_w, d_s, d_b, d_x, d_out_bf, d_out_f, n, k, grid_blocks);
    }
    chk_hip(hipDeviceSynchronize(), "q4 warmup sync");

    hipEvent_t ev0, ev1;
    chk_hip(hipEventCreate(&ev0), "ev0");
    chk_hip(hipEventCreate(&ev1), "ev1");
    chk_hip(hipEventRecord(ev0, 0), "ev0 rec");
    for (int i = 0; i < iters; ++i) {
        flint_q4_gemv_device(d_w, d_s, d_b, d_x, d_out_bf, d_out_f, n, k, grid_blocks);
    }
    chk_hip(hipEventRecord(ev1, 0), "ev1 rec");
    chk_hip(hipEventSynchronize(ev1), "ev1 sync");
    float total_ms = 0.0f;
    chk_hip(hipEventElapsedTime(&total_ms, ev0, ev1), "ev elapsed");
    if (out_ms) *out_ms = total_ms / (float)(iters > 0 ? iters : 1);
    chk_hip(hipEventDestroy(ev0), "ev0 destroy");
    chk_hip(hipEventDestroy(ev1), "ev1 destroy");

    if (h_out_bf) chk_hip(hipMemcpy(h_out_bf, d_out_bf, ob_bf, hipMemcpyDeviceToHost), "q4 D2H bf");
    if (h_out_f)  chk_hip(hipMemcpy(h_out_f, d_out_f, ob_f, hipMemcpyDeviceToHost), "q4 D2H f");

    chk_hip(hipFree(d_w), "free"); chk_hip(hipFree(d_s), "free"); chk_hip(hipFree(d_b), "free"); chk_hip(hipFree(d_x), "free");
    if (d_out_bf) chk_hip(hipFree(d_out_bf), "free");
    if (d_out_f)  chk_hip(hipFree(d_out_f), "free");
    return 0;
}

// Persistent session handle so benchmarks & multiple decode calls keep the 0.94 GiB
// mid-8L bank resident on GPU without host->device copies inside timed regions.
struct MidSession {
    uint8_t* d_bank;
    int64_t* d_off;
    bf16*    d_kv_sliding;
    bf16*    d_kv_full;
    float*   d_kv_sliding_f32;
    float*   d_kv_full_f32;
    bf16*    d_ws_bf;
    float*   d_ws_f;
    int*     d_ws_i;
    uint32_t* d_control;
    int*     d_prompt;
    int*     d_out;
    float*   d_step_logits;
    float*   d_logits;
    int*     d_chosen_experts;
    int      max_total;
};

constexpr int MID_MAX_STEP_OUT = 64;
constexpr int MID_MAX_WORKERS = 64; // R3 hard cap: ws_f_maxval has 64 slots

extern "C" int mid_query_occupancy(int mode, int* out_cus, int* out_max_blocks_per_cu,
                                   int* out_max_active_workers, int* out_wave_size) {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return -1;
    int blocks_per_cu = 0;
    const void* fn = (mode == 2) ? (const void*)mid_generate_fp32kv
                   : (mode == 1) ? (const void*)mid_generate_fp32_bf16kv
                                 : (const void*)mid_generate;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, fn, 256, 0) != hipSuccess) return -2;
    int max_active = prop.multiProcessorCount * blocks_per_cu;
    if (max_active > MID_MAX_WORKERS) max_active = MID_MAX_WORKERS;
    if (out_cus) *out_cus = prop.multiProcessorCount;
    if (out_max_blocks_per_cu) *out_max_blocks_per_cu = blocks_per_cu;
    if (out_max_active_workers) *out_max_active_workers = max_active;
    if (out_wave_size) *out_wave_size = prop.warpSize;
    return 0;
}

extern "C" void* mid_session_create(const uint8_t* h_bank, const int64_t* h_off, int max_total) {
    MidSession* s = new MidSession();
    s->max_total = max_total;
    size_t bank_sz = (size_t)h_off[MID_NUM_OFFSETS - 1];
    size_t off_sz  = MID_NUM_OFFSETS * sizeof(int64_t);
    size_t kvs_elems = (size_t)4 * 2 * MID_NKV * MID_WIN * MID_HDIM;
    size_t kvf_elems = (size_t)4 * 2 * MID_NKV * (size_t)max_total * MID_HDIM;
    size_t ws_bf_sz = 1024 * sizeof(bf16);
    size_t ws_f_sz  = (32768 + 128 + MID_VOCAB) * sizeof(float);
    size_t ws_i_sz  = 128 * sizeof(int);

    chk_hip(hipMalloc(&s->d_bank, bank_sz), "hipMalloc bank");
    chk_hip(hipMalloc(&s->d_off, off_sz), "hipMalloc off");
    chk_hip(hipMalloc(&s->d_kv_sliding, kvs_elems * sizeof(bf16)), "hipMalloc kv_sliding");
    chk_hip(hipMalloc(&s->d_kv_full, kvf_elems * sizeof(bf16)), "hipMalloc kv_full");
    chk_hip(hipMalloc(&s->d_kv_sliding_f32, kvs_elems * sizeof(float)), "hipMalloc kv_sliding_f32");
    chk_hip(hipMalloc(&s->d_kv_full_f32, kvf_elems * sizeof(float)), "hipMalloc kv_full_f32");
    chk_hip(hipMalloc(&s->d_ws_bf, ws_bf_sz), "hipMalloc ws_bf");
    chk_hip(hipMalloc(&s->d_ws_f, ws_f_sz), "hipMalloc ws_f");
    chk_hip(hipMalloc(&s->d_ws_i, ws_i_sz), "hipMalloc ws_i");
    chk_hip(hipMalloc(&s->d_control, 16 * sizeof(uint32_t)), "hipMalloc control");
    chk_hip(hipMalloc(&s->d_prompt, (size_t)max_total * sizeof(int)), "hipMalloc prompt");
    chk_hip(hipMalloc(&s->d_out, (size_t)max_total * sizeof(int)), "hipMalloc out");
    chk_hip(hipMalloc(&s->d_step_logits, (size_t)MID_MAX_STEP_OUT * MID_VOCAB * sizeof(float)), "hipMalloc step_logits");
    chk_hip(hipMalloc(&s->d_logits, (size_t)MID_VOCAB * sizeof(float)), "hipMalloc logits");
    chk_hip(hipMalloc(&s->d_chosen_experts, (size_t)max_total * MID_NL * MID_TOPK * sizeof(int)), "hipMalloc chosen_experts");

    chk_hip(hipMemcpy(s->d_bank, h_bank, bank_sz, hipMemcpyHostToDevice), "hipMemcpy bank");
    chk_hip(hipMemcpy(s->d_off, h_off, off_sz, hipMemcpyHostToDevice), "hipMemcpy off");
    return s;
}

// mode:
//   0 = production bf16-KV build (`mid_generate`, KVT=bf16, PURE_FP32=false)
//   1 = pure-fp32 compute + bf16-KV build (`mid_generate_fp32_bf16kv`, KVT=bf16, PURE_FP32=true)
//   2 = pure-fp32 compute + fp32-KV validation build (`mid_generate_fp32kv`, KVT=float, PURE_FP32=true)
// h_out_chosen_experts (optional, default nullptr):
//   int32 buffer of shape [prompt_len + n_out, MID_NL (8), MID_TOPK (2)].
//   When nullptr, costs zero device work (null pointer passed to kernel).
// Returns 0 on success (with control[2] == 0), nonzero on invalid arguments, HIP error, or barrier abort.
extern "C" int mid_session_run_ex2(void* handle, const int* h_prompt, int prompt_len,
                                   int* h_out_tokens, int n_out,
                                   float* h_out_step_logits, float* h_out_last_logits,
                                   int workers, int mode, uint32_t* out_control,
                                   int* h_out_chosen_experts) {
    MidSession* s = (MidSession*)handle;
    if (!s) return -1;
    // R3: reject invalid workers loudly (no silent default to 16; enforce compiled cap <= 64)
    if (workers < 1 || workers > MID_MAX_WORKERS) {
        fprintf(stderr, "flint_mid error: workers=%d outside valid range [1, %d]\n", workers, MID_MAX_WORKERS);
        return -2;
    }
    // A5(a): enforce full-layer KV capacity >= P + N
    if (prompt_len + n_out > s->max_total) {
        fprintf(stderr, "flint_mid error: prompt_len(%d)+n_out(%d) > max_total(%d)\n",
                prompt_len, n_out, s->max_total);
        return -3;
    }
    chk_hip(hipMemset(s->d_control, 0, 16 * sizeof(uint32_t)), "hipMemset control");
    chk_hip(hipMemcpy(s->d_prompt, h_prompt, (size_t)prompt_len * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy prompt");
    chk_hip(hipMemset(s->d_out, 0, (size_t)n_out * sizeof(int)), "hipMemset out");

    float* d_step = (h_out_step_logits && n_out <= MID_MAX_STEP_OUT) ? s->d_step_logits : nullptr;
    float* d_last = h_out_last_logits ? s->d_logits : nullptr;
    int*   d_exp  = h_out_chosen_experts ? s->d_chosen_experts : nullptr;
    if (d_exp) {
        chk_hip(hipMemset(d_exp, 0xff, (size_t)(prompt_len + n_out) * MID_NL * MID_TOPK * sizeof(int)),
                "hipMemset chosen_experts");
    }

    if (mode == 2) {
        hipLaunchKernelGGL(mid_generate_fp32kv, dim3(workers), dim3(256), 0, 0,
                           s->d_bank, s->d_off,
                           s->d_kv_sliding_f32, s->d_kv_full_f32, s->max_total,
                           s->d_prompt, prompt_len,
                           s->d_out, n_out, d_step, d_last,
                           s->d_ws_bf, s->d_ws_f, s->d_ws_i,
                           s->d_control, workers, d_exp);
    } else if (mode == 1) {
        hipLaunchKernelGGL(mid_generate_fp32_bf16kv, dim3(workers), dim3(256), 0, 0,
                           s->d_bank, s->d_off,
                           s->d_kv_sliding, s->d_kv_full, s->max_total,
                           s->d_prompt, prompt_len,
                           s->d_out, n_out, d_step, d_last,
                           s->d_ws_bf, s->d_ws_f, s->d_ws_i,
                           s->d_control, workers, d_exp);
    } else {
        hipLaunchKernelGGL(mid_generate, dim3(workers), dim3(256), 0, 0,
                           s->d_bank, s->d_off,
                           s->d_kv_sliding, s->d_kv_full, s->max_total,
                           s->d_prompt, prompt_len,
                           s->d_out, n_out, d_step, d_last,
                           s->d_ws_bf, s->d_ws_f, s->d_ws_i,
                           s->d_control, workers, d_exp);
    }
    hipError_t sync_err = hipDeviceSynchronize();
    if (sync_err != hipSuccess) {
        fprintf(stderr, "flint_mid mid_generate sync error: %s\n", hipGetErrorString(sync_err));
        return -4;
    }
    // R2 & R5: always read back control[0..2] and check control[2] (barrier abort flag)
    uint32_t h_ctrl[4] = {0, 0, 0, 0};
    chk_hip(hipMemcpy(h_ctrl, s->d_control, 4 * sizeof(uint32_t), hipMemcpyDeviceToHost), "hipMemcpy control D2H");
    if (out_control) {
        out_control[0] = h_ctrl[0];
        out_control[1] = h_ctrl[1];
        out_control[2] = h_ctrl[2];
    }
    if (h_ctrl[2] != 0u) {
        fprintf(stderr, "flint_mid BARRIER ABORT: control[2]=%u (phase=%u, arrived=%u, workers=%d)\n",
                h_ctrl[2], h_ctrl[1], h_ctrl[0], workers);
        return -5;
    }
    if (h_out_tokens && n_out > 0)
        chk_hip(hipMemcpy(h_out_tokens, s->d_out, (size_t)n_out * sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy out D2H");
    if (d_step)
        chk_hip(hipMemcpy(h_out_step_logits, s->d_step_logits, (size_t)n_out * MID_VOCAB * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy step_logits D2H");
    if (d_last)
        chk_hip(hipMemcpy(h_out_last_logits, s->d_logits, (size_t)MID_VOCAB * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy logits D2H");
    if (d_exp)
        chk_hip(hipMemcpy(h_out_chosen_experts, d_exp, (size_t)(prompt_len + n_out) * MID_NL * MID_TOPK * sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy chosen_experts D2H");
    return 0;
}

extern "C" int mid_session_run_ex(void* handle, const int* h_prompt, int prompt_len,
                                  int* h_out_tokens, int n_out,
                                  float* h_out_step_logits, float* h_out_last_logits,
                                  int workers, int mode, uint32_t* out_control) {
    return mid_session_run_ex2(handle, h_prompt, prompt_len, h_out_tokens, n_out,
                               h_out_step_logits, h_out_last_logits, workers, mode, out_control, nullptr);
}

extern "C" int mid_session_run(void* handle, const int* h_prompt, int prompt_len,
                               int* h_out_tokens, int n_out, float* h_out_last_logits,
                               int workers) {
    return mid_session_run_ex2(handle, h_prompt, prompt_len, h_out_tokens, n_out, nullptr, h_out_last_logits, workers, 0, nullptr, nullptr);
}

extern "C" void mid_session_free(void* handle) {
    MidSession* s = (MidSession*)handle;
    if (!s) return;
    chk_hip(hipFree(s->d_bank), "hipFree");
    chk_hip(hipFree(s->d_off), "hipFree");
    chk_hip(hipFree(s->d_kv_sliding), "hipFree");
    chk_hip(hipFree(s->d_kv_full), "hipFree");
    chk_hip(hipFree(s->d_kv_sliding_f32), "hipFree");
    chk_hip(hipFree(s->d_kv_full_f32), "hipFree");
    chk_hip(hipFree(s->d_ws_bf), "hipFree");
    chk_hip(hipFree(s->d_ws_f), "hipFree");
    chk_hip(hipFree(s->d_ws_i), "hipFree");
    chk_hip(hipFree(s->d_control), "hipFree");
    chk_hip(hipFree(s->d_prompt), "hipFree");
    chk_hip(hipFree(s->d_out), "hipFree");
    chk_hip(hipFree(s->d_step_logits), "hipFree");
    chk_hip(hipFree(s->d_logits), "hipFree");
    chk_hip(hipFree(s->d_chosen_experts), "hipFree");
    delete s;
}
