// Flint tiny-config MEGAKERNEL: one persistent kernel generates a full
// greedy decode (whole model, zero host launches per token).
//
// Tiny config = synthetic checkpoint: hidden 128, 4 q-heads / 2 kv-heads,
// head_dim 64, 2 layers (L0 dense MLP, L1 MoE 8 experts top-2 + shared),
// vocab 512, L0 rotary / L1 not, per-q-head attention SINKS, greedy decode.
//
// Design notes (validation phase — parallelize later):
//  - hidden state lives in SHARED memory (q4/rmsnorm index lane*4.., so all
//    phase inputs must be shared, never per-lane registers);
//  - q4 matmul: K templated, lane covers K/32 consecutive elements,
//    literal MSL nibble unpack + x-scaling bias fold;
//  - head_norm+RoPE runs on lane 0 only (serial loop would race across lanes);
//  - attention: streaming softmax with per-head sinks, serial dims,
//    one lane per kv group looping its q-heads.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdio>

typedef __hip_bfloat16 bf16;

constexpr int HID = 128;
constexpr int HEADS = 4;
constexpr int NKV = 2;
constexpr int HDIM = 64;
constexpr int NEXP = 8;
constexpr int TOPK = 2;
constexpr int MOE_I = 64;
constexpr int SH_I = 64;
constexpr int DENSE_I = 256;
constexpr int VOCAB = 512;
constexpr int NL = 2;
constexpr float EPS = 1e-6f;
constexpr float ATTN_SCALE = 0.125f;               // 1/sqrt(64)
constexpr float ROPE_LOGBASE = 13.28771237954945f; // log2(10000)

// ---- bf16 tensor table ----
constexpr int IDX_EMBED = 0, IDX_L0LN = 1, IDX_L0POST = 2, IDX_L0QN = 3, IDX_L0KN = 4,
    IDX_L0SINKS = 5, IDX_L1LN = 6, IDX_L1POST = 7, IDX_L1QN = 8, IDX_L1KN = 9,
    IDX_L1SINKS = 10, IDX_ROUTER_W = 11, IDX_ROUTER_B = 12, IDX_FNNORM = 13;
constexpr int NBF16 = 14;
[[maybe_unused]] constexpr int BF16_ELEMS[NBF16] = {
    VOCAB * HID, HID, HID, HDIM, HDIM, HEADS,
    HID, HID, HDIM, HDIM, HEADS,
    NEXP * HID, NEXP, HID,
};

// ---- q4 tensor table ----
constexpr int Q_L0 = 0, K_L0 = 1, V_L0 = 2, O_L0 = 3,
    GATE_L0 = 4, UP_L0 = 5, DOWN_L0 = 6,
    Q_L1 = 7, K_L1 = 8, V_L1 = 9, O_L1 = 10,
    SW_GATE = 11, SW_UP = 12, SW_DOWN = 13,
    SH_GATE = 14, SH_UP = 15, SH_DOWN = 16, LM_HEAD = 17;
constexpr int NQ4 = 18;
constexpr int QN[NQ4] = {
    HEADS * HDIM, NKV * HDIM, NKV * HDIM, HID,          // L0 q k v o
    DENSE_I, DENSE_I, HID,                              // L0 gate up down
    HEADS * HDIM, NKV * HDIM, NKV * HDIM, HID,          // L1 q k v o
    NEXP * MOE_I, NEXP * MOE_I, NEXP * HID,             // L1 sw gate up down
    SH_I, SH_I, HID,                                    // L1 sh gate up down
    VOCAB,                                              // lm head
};
constexpr int QK[NQ4] = {
    HID, HID, HID, DENSE_I,
    HID, HID, DENSE_I,
    HID, HID, HID, DENSE_I,
    HID, HID, MOE_I,
    HID, HID, MOE_I,
    HID,
};

constexpr int Q4_FIRST = NBF16;   // q4 tensors start after the bf16 block

constexpr int q4_packed_bytes(int idx) { return QN[idx] * QK[idx] / 2; }
constexpr int q4_sb_bytes(int idx) { return QN[idx] * (QK[idx] / 64) * 2; }
constexpr int q4_total_bytes(int idx) { return q4_packed_bytes(idx) + 2 * q4_sb_bytes(idx); }

__device__ inline float tofloat(bf16 v) { return __bfloat162float(v); }
__device__ inline bf16 tobf(float v) { return __float2bfloat16(v); }

__device__ inline float silu(float v) { return v / (1.0f + expf(-v)); }

__device__ inline float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        v += __shfl_xor_sync(0xffffffffffffffffull, v, off);
    return v;
}

constexpr uint16_t NIB[4] = {15u, 240u, 3840u, 61440u};

// q4 matmul, compile-time K (K % 32 == 0): lane covers K/32 consecutive
// K-elements = (K/32)/4 ushorts. x: shared K floats; out: float[n] zeroed by
// caller. Literal MSL nibble arithmetic with the x-scaling bias fold.
// Multi-warp + multi-workgroup: global warp `sg` steps by `groups * 4` rows.
template <int K, int TAG = 0>
__device__ inline void q4(const uint8_t* w, const bf16* s, const bf16* b,
                          const float* x, float* out, int n,
                          int sg = 0, int groups = 1) {
    int lane = threadIdx.x & 31;
#ifdef DEBUG_TINY
    if (TAG == 9 && lane == 0 && sg == 0)
        printf("MK q4ptr t9 w %llu s %llu b %llu\n",
               (unsigned long long)w, (unsigned long long)s, (unsigned long long)b);
#endif
    constexpr int PER = K / 32;
    constexpr int NU = PER / 4;
    int scalar = lane * PER;
    float v[PER], vs[PER], sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < PER; ++j) { v[j] = x[scalar + j]; sum += v[j]; }
    #pragma unroll
    // MSL load_vector pre-scale: period 4 (one ushort = 4 nibbles), i.e.
    // x/1, x/16, x/256, x/4096 repeating. The old exponent (j>>1)*2+(j&1) == j
    // scaled elements 4..7 of K=256 rows by 16^-4 (o_proj, dense down).
    for (int j = 0; j < PER; ++j) vs[j] = v[j] / (float)(1u << (4 * (j & 3)));
    for (int first = sg * 4; first < n; first += groups * 4) {
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int r = 0; r < 4; ++r) {
            int row = first + r;
            if (row >= n) break;
            int gi = row * (K / 64) + scalar / 64;
            float sc = tofloat(s[gi]), bi = tofloat(b[gi]);
            const uint16_t* p = (const uint16_t*)(w + (size_t)row * (K / 2) + scalar / 2);
            float dsum = 0.0f;
            if constexpr (NU > 0) {
                #pragma unroll
                for (int u = 0; u < NU; ++u)
                    #pragma unroll
                    for (int nib = 0; nib < 4; ++nib) {
                        int j = u * 4 + nib;
                        dsum += vs[j] * (p[u] & NIB[nib]);
                    }
            } else {
                // PER == 2 (K % 128 != 0): lane owns 2 elements = 1 byte
                static_assert(PER == 2, "q4: K must satisfy K % 128 == 0 or K % 64 == 0");
                const uint8_t* pb = w + (size_t)row * (K / 2) + lane;
                dsum = vs[0] * (pb[0] & 15u) + vs[1] * (pb[0] & 240u);
            }
            a[r] += sc * dsum + sum * bi;
        }
        for (int r = 0; r < 4; ++r) {
            float v = warp_sum(a[r]);
            if (lane == 0 && first + r < n) out[first + r] += v;
        }
    }
}

// Device-wide spin barrier (MSL sync_all equivalent) across `workers` workgroups.
// control[0]: arrived count, control[1]: current phase, control[2]: abort flag.
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
               spins < 20000000u) {
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

// RMSNorm over HID from shared in[] to shared out[]; warp 0 owns the 128 elements (4/lane).
__device__ inline void rmsnorm(const float* in, const bf16* w, float* out) {
    if ((threadIdx.x >> 5) != 0) return;
    int lane = threadIdx.x & 31;
    float squared = 0.0f;
    for (int j = 0; j < 4; ++j) { float v = in[lane * 4 + j]; squared += v * v; }
    float inv = 1.0f / sqrtf(warp_sum(squared) / (float)HID + EPS);
    for (int j = 0; j < 4; ++j) {
        int i = lane * 4 + j;
        out[i] = tofloat(w[i]) * in[i] * inv;
    }
}

// per-head RMSNorm (+ optional NeoX-half RoPE) over a shared q/k buffer.
// Lane 0 only: the serial loop reads and rewrites the same elements.
__device__ inline void head_norm_rope(float* qk, const bf16* nw, int heads,
                                      int rotary, int position) {
    if (threadIdx.x != 0) return;
    for (int h = 0; h < heads; ++h) {
        float* p = qk + (size_t)h * HDIM;
        float sq = 0.0f;
        for (int d = 0; d < HDIM; ++d) sq += p[d] * p[d];
        float inv = 1.0f / sqrtf(sq / (float)HDIM + EPS);
        for (int d = 0; d < HDIM; ++d) p[d] = tofloat(nw[d]) * p[d] * inv;
        if (rotary) {
            for (int pair = 0; pair < HDIM / 2; ++pair) {
                float a = p[pair], b = p[pair + HDIM / 2];
                float theta = (float)position * exp2f(-(float)pair / (float)(HDIM / 2) * ROPE_LOGBASE);
                float cc = cosf(theta), ss = sinf(theta);
                p[pair] = a * cc - b * ss;
                p[pair + HDIM / 2] = a * ss + b * cc;
            }
        }
    }
}

extern "C" __global__ void tiny_generate(
    const uint8_t* bank, const int64_t* off,
    bf16* kv,                                // [NL][2 (K|V)][NKV][total][HDIM]
    const int* prompt, int prompt_len,
    int* out_tokens, int n_out, float* out_last_logits,
    float* ws, uint32_t* control, int workers) {
    const int tid = threadIdx.x;
    const int group = blockIdx.x;
    const int bthreads = blockDim.x;
    const int bwarps = bthreads >> 5;
    const int sg = group * bwarps + (tid >> 5);
    const int groups = workers * bwarps;
    const int total = prompt_len + n_out;
    const size_t cap = (size_t)total;

    auto OFF = [&](int idx) -> int64_t { return off[idx]; };
    auto B = [&](int idx) -> const bf16* { return (const bf16*)(bank + OFF(idx)); };
    // q4 tensor indices are local to the q4 table — add Q4_FIRST for the bank
    auto QW = [&](int idx) -> const uint8_t* { return bank + OFF(Q4_FIRST + idx); };
    auto QS = [&](int idx) -> const bf16* {
        return (const bf16*)(bank + OFF(Q4_FIRST + idx) + q4_packed_bytes(idx)); };
    auto QB = [&](int idx) -> const bf16* {
        return (const bf16*)(bank + OFF(Q4_FIRST + idx) + q4_packed_bytes(idx) + q4_sb_bytes(idx)); };

    // When workers > 1, output buffers for multi-workgroup phases live in device workspace `ws`
    // so all workgroups write to and read from the same arrays across `sync_all`.
    __shared__ float sh_x[HID];       // residual stream (pre-norm, replicated per workgroup)
    __shared__ float sh_xn[HID];      // normalized copy for q4 inputs (replicated per workgroup)
    __shared__ float sh_q_local[HEADS * HDIM], sh_k_local[NKV * HDIM], sh_v_local[NKV * HDIM];
    __shared__ float sh_attn_local[HEADS * HDIM], sh_o_local[HID], sh_mlp_local[HID];
    __shared__ float sh_mid_local[DENSE_I], sh_up_local[DENSE_I];
    __shared__ float sh_score[NEXP], sh_wsel[TOPK], sh_logits_local[VOCAB];
    __shared__ int sh_sel[TOPK];
    __shared__ uint32_t sh_ready[1];
    uint32_t phase = 0;

    float* sh_q      = ws ? (ws + 0) : sh_q_local;
    float* sh_k      = ws ? (ws + HEADS * HDIM) : sh_k_local;
    float* sh_v      = ws ? (ws + (HEADS + NKV) * HDIM) : sh_v_local;
    float* sh_attn   = ws ? (ws + (HEADS + 2 * NKV) * HDIM) : sh_attn_local;
    float* sh_o      = ws ? (ws + (2 * HEADS + 2 * NKV) * HDIM) : sh_o_local;
    float* sh_mlp    = ws ? (ws + (2 * HEADS + 2 * NKV) * HDIM + HID) : sh_mlp_local;
    float* sh_mid    = ws ? (ws + (2 * HEADS + 2 * NKV) * HDIM + 2 * HID) : sh_mid_local;
    float* sh_up     = ws ? (ws + (2 * HEADS + 2 * NKV) * HDIM + 2 * HID + DENSE_I) : sh_up_local;
    float* sh_logits = ws ? (ws + (2 * HEADS + 2 * NKV) * HDIM + 2 * HID + 2 * DENSE_I) : sh_logits_local;
    float* ws_moe    = ws ? (ws + (2 * HEADS + 2 * NKV) * HDIM + 2 * HID + 2 * DENSE_I + VOCAB) : nullptr;

    #define SYNC_ALL() if (!sync_all(control, tid, phase, sh_ready, workers)) return

    for (int pos = 0; pos < total; ++pos) {
        int token = (pos < prompt_len) ? prompt[pos] : out_tokens[pos - prompt_len];

        // ---- embedding (replicated across workgroups) ----
        for (int i = tid; i < HID; i += bthreads)
            sh_x[i] = tofloat(B(IDX_EMBED)[(size_t)token * HID + i]);
        __syncthreads();

        for (int layer = 0; layer < NL; ++layer) {
            int ln = layer == 0 ? IDX_L0LN : IDX_L1LN;
            int post = layer == 0 ? IDX_L0POST : IDX_L1POST;
            int qni = layer == 0 ? IDX_L0QN : IDX_L1QN;
            int kni = layer == 0 ? IDX_L0KN : IDX_L1KN;
            int sinks = layer == 0 ? IDX_L0SINKS : IDX_L1SINKS;
            int qidx = layer == 0 ? Q_L0 : Q_L1;
            int kidx = layer == 0 ? K_L0 : K_L1;
            int vidx = layer == 0 ? V_L0 : V_L1;
            int oidx = layer == 0 ? O_L0 : O_L1;
            int rotary = layer == 0;

            // ---- input RMSNorm -> sh_xn ----
            rmsnorm(sh_x, B(ln), sh_xn);
            __syncthreads();

            // ---- qkv (all K = HID) ----
            if (group == 0) {
                for (int i = tid; i < HEADS * HDIM; i += bthreads) sh_q[i] = 0.0f;
                for (int i = tid; i < NKV * HDIM; i += bthreads) { sh_k[i] = 0.0f; sh_v[i] = 0.0f; }
                for (int i = tid; i < HEADS * HDIM; i += bthreads) sh_attn[i] = 0.0f;
                for (int i = tid; i < HID; i += bthreads) sh_o[i] = 0.0f;
            }
            SYNC_ALL();
            q4<HID, 1>(QW(qidx), QS(qidx), QB(qidx), sh_xn, sh_q, HEADS * HDIM, sg, groups);
            q4<HID, 2>(QW(kidx), QS(kidx), QB(kidx), sh_xn, sh_k, NKV * HDIM, sg, groups);
            q4<HID, 3>(QW(vidx), QS(vidx), QB(vidx), sh_xn, sh_v, NKV * HDIM, sg, groups);
            SYNC_ALL();

            // ---- head norm + rope on q and k ----
            if (group == 0) {
                head_norm_rope(sh_q, B(qni), HEADS, rotary, pos);
                head_norm_rope(sh_k, B(kni), NKV, rotary, pos);
            }
            SYNC_ALL();

            // ---- write k, v into cache at pos ----
            if (group == 0) {
                for (int h = 0; h < NKV; ++h)
                    for (int d = tid; d < HDIM; d += bthreads) {
                        // Metal KVStorage head-major order: [NL][2 (K|V)][NKV][cap][HDIM]
                        size_t kk = (((size_t)layer * 2 + 0) * NKV + h) * cap * HDIM + (size_t)pos * HDIM + d;
                        size_t vv = (((size_t)layer * 2 + 1) * NKV + h) * cap * HDIM + (size_t)pos * HDIM + d;
                        kv[kk] = tobf(sh_k[h * HDIM + d]);
                        kv[vv] = tobf(sh_v[h * HDIM + d]);
                    }
            }
            SYNC_ALL();

            // ---- attention with sinks ----
            const bf16* sinkp = B(sinks);
            int nqh = HEADS / NKV;
            if (group == 0 && tid < nqh) {
                int kvh = tid;
                const bf16* kc = kv + (((size_t)layer * 2 + 0) * NKV + kvh) * cap * HDIM;
                const bf16* vc = kv + (((size_t)layer * 2 + 1) * NKV + kvh) * cap * HDIM;
                for (int g = 0; g < nqh; ++g) {
                    int h = kvh * nqh + g;
                    float hi = tofloat(sinkp[h]);
                    float z = 1.0f;
                    float qreg[HDIM], acc[HDIM];
                    for (int d = 0; d < HDIM; ++d) {
                        qreg[d] = tofloat(sh_q[h * HDIM + d]) * ATTN_SCALE;
                        acc[d] = 0.0f;
                    }
                    for (int t = 0; t <= pos; ++t) {
                        float score = 0.0f;
                        for (int d = 0; d < HDIM; ++d) score += qreg[d] * tofloat(kc[(size_t)t * HDIM + d]);
                        float next = fmaxf(hi, score);
                        float low = expf(-fabsf(score - hi));
                        bool wins = score > hi;
                        float old = wins ? low : 1.0f, add = wins ? 1.0f : low;
                        for (int d = 0; d < HDIM; ++d)
                            acc[d] = acc[d] * old + add * tofloat(vc[(size_t)t * HDIM + d]);
                        z = z * old + add;
                        hi = next;
                    }
                    for (int d = 0; d < HDIM; ++d) sh_attn[h * HDIM + d] = acc[d] / z;
                }
            }
            SYNC_ALL();

            // ---- o proj + residual ----
            q4<DENSE_I>(QW(oidx), QS(oidx), QB(oidx), sh_attn, sh_o, HID, sg, groups);
            SYNC_ALL();
            for (int i = tid; i < HID; i += bthreads) sh_x[i] += sh_o[i];
            __syncthreads();

            // ---- post RMSNorm -> sh_xn ----
            rmsnorm(sh_x, B(post), sh_xn);
            __syncthreads();

            // ---- MLP / MoE ----
            if (layer == 0) {
                if (group == 0) {
                    for (int i = tid; i < DENSE_I; i += bthreads) { sh_mid[i] = 0.0f; sh_up[i] = 0.0f; }
                    for (int i = tid; i < HID; i += bthreads) sh_mlp[i] = 0.0f;
                }
                SYNC_ALL();
                q4<HID, 7>(QW(GATE_L0), QS(GATE_L0), QB(GATE_L0), sh_xn, sh_mid, DENSE_I, sg, groups);
                q4<HID, 8>(QW(UP_L0), QS(UP_L0), QB(UP_L0), sh_xn, sh_up, DENSE_I, sg, groups);
                SYNC_ALL();
                if (group == 0) {
                    for (int i = tid; i < DENSE_I; i += bthreads)
                        sh_mid[i] = tobf(silu(tofloat(tobf(sh_mid[i]))) * tofloat(sh_up[i]));
                }
                SYNC_ALL();
                q4<DENSE_I>(QW(DOWN_L0), QS(DOWN_L0), QB(DOWN_L0), sh_mid, sh_mlp, HID, sg, groups);
                SYNC_ALL();
                for (int i = tid; i < HID; i += bthreads) sh_x[i] += sh_mlp[i];
                __syncthreads();
            } else {
                if (tid == 0) {
                    int top[TOPK]; float tv[TOPK];
                    for (int t = 0; t < TOPK; ++t) { tv[t] = -1e30f; top[t] = 0; }
                    const bf16* rw = B(IDX_ROUTER_W);
                    for (int e = 0; e < NEXP; ++e) {
                        float sc = 0.0f;
                        for (int i = 0; i < HID; ++i) sc += tofloat(rw[(size_t)e * HID + i]) * sh_xn[i];
                        float sig = 1.0f / (1.0f + expf(-tofloat(tobf(sc))));
                        float choice = sig + tofloat(B(IDX_ROUTER_B)[e]);
                        sh_score[e] = sig;
                        for (int t = 0; t < TOPK; ++t)
                            if (choice > tv[t]) {
                                for (int u = TOPK - 1; u > t; --u) { tv[u] = tv[u - 1]; top[u] = top[u - 1]; }
                                tv[t] = choice; top[t] = e;
                                break;
                            }
                    }
                    for (int t = 0; t < TOPK; ++t) sh_sel[t] = top[t];
                    float s0 = sh_score[sh_sel[0]], s1 = sh_score[sh_sel[1]];
                    float wsum = s0 + s1 + 1e-20f;
                    sh_wsel[0] = s0 / wsum; sh_wsel[1] = s1 / wsum;
                }
                __syncthreads();
                if (group == 0) {
                    for (int i = tid; i < HID; i += bthreads) sh_mlp[i] = 0.0f;
                }
                SYNC_ALL();
                __shared__ float sh_mi_local[MOE_I], sh_u2_local[MOE_I], sh_dyn_local[HID];
                float* sh_mi  = ws_moe ? (ws_moe + 0) : sh_mi_local;
                float* sh_u2  = ws_moe ? (ws_moe + MOE_I) : sh_u2_local;
                float* sh_dyn = ws_moe ? (ws_moe + 2 * MOE_I) : sh_dyn_local;
                for (int t = 0; t < TOPK; ++t) {
                    int e = sh_sel[t];
                    if (group == 0) {
                        for (int i = tid; i < MOE_I; i += bthreads) { sh_mi[i] = 0.0f; sh_u2[i] = 0.0f; }
                        for (int i = tid; i < HID; i += bthreads) sh_dyn[i] = 0.0f;
                    }
                    SYNC_ALL();
                    q4<HID, 9>(QW(SW_GATE) + (size_t)e * (MOE_I * HID / 2),
                            QS(SW_GATE) + (size_t)e * (MOE_I * (HID / 64)),
                            QB(SW_GATE) + (size_t)e * (MOE_I * (HID / 64)),
                            sh_xn, sh_mi, MOE_I, sg, groups);
                    q4<HID>(QW(SW_UP) + (size_t)e * (MOE_I * HID / 2),
                            QS(SW_UP) + (size_t)e * (MOE_I * (HID / 64)),
                            QB(SW_UP) + (size_t)e * (MOE_I * (HID / 64)),
                            sh_xn, sh_u2, MOE_I, sg, groups);
                    SYNC_ALL();
                    if (group == 0) {
                        for (int i = tid; i < MOE_I; i += bthreads)
                            sh_mi[i] = tobf(silu(tofloat(tobf(sh_mi[i]))) * tofloat(sh_u2[i]));
                    }
                    SYNC_ALL();
                    q4<MOE_I>(QW(SW_DOWN) + (size_t)e * (HID * MOE_I / 2),
                              QS(SW_DOWN) + (size_t)e * (HID * (MOE_I / 64)),
                              QB(SW_DOWN) + (size_t)e * (HID * (MOE_I / 64)),
                              sh_mi, sh_dyn, HID, sg, groups);
                    SYNC_ALL();
                    if (group == 0) {
                        for (int i = tid; i < HID; i += bthreads) sh_mlp[i] += sh_wsel[t] * sh_dyn[i];
                    }
                    SYNC_ALL();
                }
                // shared expert
                __shared__ float sh_sg_local[SH_I], sh_su_local[SH_I], sh_sd_local[HID];
                float* sh_sg = ws_moe ? (ws_moe + 2 * MOE_I + HID) : sh_sg_local;
                float* sh_su = ws_moe ? (ws_moe + 2 * MOE_I + HID + SH_I) : sh_su_local;
                float* sh_sd = ws_moe ? (ws_moe + 2 * MOE_I + HID + 2 * SH_I) : sh_sd_local;
                if (group == 0) {
                    for (int i = tid; i < SH_I; i += bthreads) { sh_sg[i] = 0.0f; sh_su[i] = 0.0f; }
                    for (int i = tid; i < HID; i += bthreads) sh_sd[i] = 0.0f;
                    for (int i = tid; i < VOCAB; i += bthreads) sh_logits[i] = 0.0f;
                }
                SYNC_ALL();
                q4<HID>(QW(SH_GATE), QS(SH_GATE), QB(SH_GATE), sh_xn, sh_sg, SH_I, sg, groups);
                q4<HID>(QW(SH_UP), QS(SH_UP), QB(SH_UP), sh_xn, sh_su, SH_I, sg, groups);
                SYNC_ALL();
                if (group == 0) {
                    for (int i = tid; i < SH_I; i += bthreads)
                        sh_sg[i] = tobf(silu(tofloat(tobf(sh_sg[i]))) * tofloat(sh_su[i]));
                }
                SYNC_ALL();
                q4<SH_I>(QW(SH_DOWN), QS(SH_DOWN), QB(SH_DOWN), sh_sg, sh_sd, HID, sg, groups);
                SYNC_ALL();
                if (group == 0) {
                    for (int i = tid; i < HID; i += bthreads) sh_mlp[i] += sh_sd[i];
                }
                SYNC_ALL();
                for (int i = tid; i < HID; i += bthreads) sh_x[i] += sh_mlp[i];
                __syncthreads();
            }
        }

        // ---- final norm + lm head + greedy argmax ----
        rmsnorm(sh_x, B(IDX_FNNORM), sh_xn);
        __syncthreads();
        q4<HID>(QW(LM_HEAD), QS(LM_HEAD), QB(LM_HEAD), sh_xn, sh_logits, VOCAB, sg, groups);
        SYNC_ALL();
        if (group == 0 && tid == 0) {
            int best = 0;
            for (int i = 1; i < VOCAB; ++i) if (sh_logits[i] > sh_logits[best]) best = i;
            if (out_last_logits) for (int i = 0; i < VOCAB; ++i) out_last_logits[i] = sh_logits[i];
            int slot = pos - prompt_len + 1;
            if (slot >= 0 && slot < n_out) out_tokens[slot] = best;
        }
        SYNC_ALL();
    }
    #undef SYNC_ALL
}

extern "C" void tiny_launch_ex(const uint8_t* h_bank, const int64_t* h_off, bf16* h_kv,
                               const int* h_prompt, int prompt_len, int* h_out_tokens,
                               int n_out, float* h_out_last_logits,
                               int workers, int block_threads) {
    if (workers < 1) workers = 1;
    if (block_threads < 32) block_threads = 32;
    int total = prompt_len + n_out;
    size_t bank_sz = (size_t)h_off[31] + q4_total_bytes(17);
    size_t off_sz = 32 * sizeof(int64_t);
    size_t kv_sz = (size_t)NL * NKV * 2 * (size_t)total * HDIM * sizeof(bf16);
    size_t ws_elems = (2 * HEADS + 2 * NKV) * HDIM + 2 * HID + 2 * DENSE_I + VOCAB + 2 * MOE_I + 2 * HID + 2 * SH_I;

    uint8_t* d_bank; int64_t* d_off; bf16* d_kv; int* d_prompt; int* d_out; float* d_logits;
    float* d_ws = nullptr; uint32_t* d_control = nullptr;
    auto chk = [](hipError_t e, const char* what) {
        if (e != hipSuccess) fprintf(stderr, "tiny_launch %s: %s\n", what, hipGetErrorString(e));
    };
    chk(hipMalloc(&d_bank, bank_sz), "hipMalloc bank");
    chk(hipMalloc(&d_off, off_sz), "hipMalloc off");
    chk(hipMalloc(&d_kv, kv_sz), "hipMalloc kv");
    chk(hipMalloc(&d_prompt, (size_t)prompt_len * sizeof(int)), "hipMalloc prompt");
    chk(hipMalloc(&d_out, (size_t)n_out * sizeof(int)), "hipMalloc out");
    chk(hipMalloc(&d_logits, h_out_last_logits ? (size_t)VOCAB * sizeof(float) : 0), "hipMalloc logits");
    chk(hipMalloc(&d_ws, ws_elems * sizeof(float)), "hipMalloc ws");
    chk(hipMalloc(&d_control, 16 * sizeof(uint32_t)), "hipMalloc control");
    chk(hipMemset(d_ws, 0, ws_elems * sizeof(float)), "hipMemset ws");
    chk(hipMemset(d_control, 0, 16 * sizeof(uint32_t)), "hipMemset control");
    chk(hipMemcpy(d_bank, h_bank, bank_sz, hipMemcpyHostToDevice), "hipMemcpy bank");
    chk(hipMemcpy(d_off, h_off, off_sz, hipMemcpyHostToDevice), "hipMemcpy off");
    chk(hipMemcpy(d_kv, h_kv, kv_sz, hipMemcpyHostToDevice), "hipMemcpy kv");
    chk(hipMemcpy(d_prompt, h_prompt, (size_t)prompt_len * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy prompt");
    chk(hipMemcpy(d_out, h_out_tokens, (size_t)n_out * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy out");
    if (h_out_last_logits) chk(hipMemcpy(d_logits, h_out_last_logits, VOCAB * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy logits");

    hipLaunchKernelGGL(tiny_generate, dim3(workers), dim3(block_threads), 0, 0,
                       d_bank, d_off, d_kv, d_prompt, prompt_len, d_out, n_out, d_logits,
                       d_ws, d_control, workers);
    chk(hipDeviceSynchronize(), "sync");
    chk(hipMemcpy(h_out_tokens, d_out, (size_t)n_out * sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy out D2H");
    if (h_out_last_logits) chk(hipMemcpy(h_out_last_logits, d_logits, VOCAB * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy logits D2H");
    chk(hipFree(d_bank), "hipFree"); chk(hipFree(d_off), "hipFree"); chk(hipFree(d_kv), "hipFree");
    chk(hipFree(d_prompt), "hipFree"); chk(hipFree(d_out), "hipFree"); chk(hipFree(d_logits), "hipFree");
    chk(hipFree(d_ws), "hipFree ws"); chk(hipFree(d_control), "hipFree control");
}

extern "C" void tiny_launch(const uint8_t* h_bank, const int64_t* h_off, bf16* h_kv,
                            const int* h_prompt, int prompt_len, int* h_out_tokens,
                            int n_out, float* h_out_last_logits) {
    tiny_launch_ex(h_bank, h_off, h_kv, h_prompt, prompt_len, h_out_tokens, n_out, h_out_last_logits, 4, 128);
}
