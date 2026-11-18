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
constexpr int BF16_ELEMS[NBF16] = {
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
template <int K, int TAG = 0>
__device__ inline void q4(const uint8_t* w, const bf16* s, const bf16* b,
                          const float* x, float* out, int n) {
    int lane = threadIdx.x & 31;
#ifdef DEBUG_TINY
    if (TAG == 9 && lane == 0)
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
    for (int j = 0; j < PER; ++j) vs[j] = v[j] / (float)(1u << (4 * ((j >> 1) * 2 + (j & 1))));
    for (int first = 0; first < n; first += 4) {
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int r = 0; r < 4; ++r) {
            int row = first + r;
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
#ifdef DEBUG_TINY
            if (TAG != 0 && row < 2 && threadIdx.x == 0)
                printf("MK q4 t%d row%d sc %.6f bi %.6f dsum %.6f sum %.6f a %.6f\n",
                       TAG, row, sc, bi, dsum, sum, a[r]);
#ifdef DEBUG_TINY
            if (TAG == 8 && row == 1)
                printf("MK upL lane %d a %.6e sc %.6e dsum %.6e\n", (int)lane, a[r], sc, dsum);
#endif
#endif
        }
        for (int r = 0; r < 4; ++r) {
            float v = warp_sum(a[r]);
            if (lane == 0) out[first + r] += v;
        }
    }
}

// RMSNorm over HID from shared in[] to shared out[]; lane owns 4 elements.
__device__ inline void rmsnorm(const float* in, const bf16* w, float* out) {
    int lane = threadIdx.x;
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
    bf16* kv,                                // [NL][NKV][2][total][HDIM]
    const int* prompt, int prompt_len,
    int* out_tokens, int n_out, float* out_last_logits) {
    const int lane = threadIdx.x & 31;
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

    // shared scratch — the whole working set of the tiny model
    __shared__ float sh_x[HID];       // residual stream (pre-norm)
    __shared__ float sh_xn[HID];      // normalized copy for q4 inputs
    __shared__ float sh_q[HEADS * HDIM], sh_k[NKV * HDIM], sh_v[NKV * HDIM];
    __shared__ float sh_attn[HEADS * HDIM], sh_o[HID], sh_mlp[HID];
    __shared__ float sh_mid[DENSE_I]; // widest intermediate (L0 MLP)
    __shared__ float sh_up[DENSE_I];  // L0 up-proj output (256 wide)
    __shared__ float sh_score[NEXP], sh_wsel[TOPK], sh_logits[VOCAB];
    __shared__ int sh_sel[TOPK], sh_next;

    for (int pos = 0; pos < total; ++pos) {
        int token = (pos < prompt_len) ? prompt[pos] : out_tokens[pos - prompt_len];

        // ---- embedding ----
        for (int i = lane; i < HID; i += 32)
            sh_x[i] = tofloat(B(IDX_EMBED)[(size_t)token * HID + i]);
#ifdef DEBUG_TINY
        if (pos == 0 && lane == 0) printf("MK embed %.6f %.6f %.6f %.6f\n", sh_x[0], sh_x[1], sh_x[2], sh_x[3]);
#endif

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
#ifdef DEBUG_TINY
            if (pos == 0 && lane == 0) printf("MK after-rmsnorm %.6f %.6f %.6f %.6f\n", sh_xn[0], sh_xn[1], sh_xn[2], sh_xn[3]);
#endif

            // ---- qkv (all K = HID) ----
            for (int i = lane; i < HEADS * HDIM; i += 32) sh_q[i] = 0.0f;
            for (int i = lane; i < NKV * HDIM; i += 32) { sh_k[i] = 0.0f; sh_v[i] = 0.0f; }
            __syncthreads();
            q4<HID, 1>(QW(qidx), QS(qidx), QB(qidx), sh_xn, sh_q, HEADS * HDIM);
#ifdef DEBUG_TINY
            if (pos == 0 && layer == 0 && lane == 0) printf("MK qk-after-q k0 %.6f v0 %.6f\n", sh_k[0], sh_v[0]);
#endif
            q4<HID, 2>(QW(kidx), QS(kidx), QB(kidx), sh_xn, sh_k, NKV * HDIM);
#ifdef DEBUG_TINY
            if (pos == 0 && layer == 0 && lane == 0) printf("MK qk-after-k k0 %.6f v0 %.6f\n", sh_k[0], sh_v[0]);
#endif
            q4<HID, 3>(QW(vidx), QS(vidx), QB(vidx), sh_xn, sh_v, NKV * HDIM);
#ifdef DEBUG_TINY
            if (pos == 0 && layer == 0 && lane == 0) printf("MK qk-after-v k0 %.6f v0 %.6f\n", sh_k[0], sh_v[0]);
#endif
            __syncthreads();

            // ---- head norm + rope on q and k ----
#ifdef DEBUG_TINY
            if (pos == 0 && lane == 0) printf("MK after-qkv-q %.6f %.6f %.6f %.6f\n", sh_q[0], sh_q[1], sh_q[2], sh_q[3]);
#endif
            head_norm_rope(sh_q, B(qni), HEADS, rotary, pos);
            head_norm_rope(sh_k, B(kni), NKV, rotary, pos);
            __syncthreads();

            // ---- write k, v into cache at pos ----
#ifdef DEBUG_TINY
            if (pos == 0 && lane == 0) printf("MK after-headnorm-q %.6f %.6f %.6f %.6f\n", sh_q[0], sh_q[1], sh_q[2], sh_q[3]);
#endif
            for (int h = 0; h < NKV; ++h)
                for (int d = lane; d < HDIM; d += 32) {
                    size_t kk = (((size_t)layer * NKV + h) * 2 + 0) * cap * HDIM + (size_t)pos * HDIM + d;
                    size_t vv = (((size_t)layer * NKV + h) * 2 + 1) * cap * HDIM + (size_t)pos * HDIM + d;
                    kv[kk] = tobf(sh_k[h * HDIM + d]);
                    kv[vv] = tobf(sh_v[h * HDIM + d]);
                }
            __syncthreads();
#ifdef DEBUG_TINY
            if (pos == 0 && layer == 0 && lane == 0)
                printf("MK kvwrite sh_k0 %.6f sh_v0 %.6f kvk0 %.6f kvv0 %.6f\\n",
                       sh_k[0], sh_v[0], tofloat(kv[0]), tofloat(kv[((size_t)NKV * 2) * cap * HDIM]));
#endif

            // ---- attention with sinks ----
            const bf16* sinkp = B(sinks);
            for (int i = lane; i < HEADS * HDIM; i += 32) sh_attn[i] = 0.0f;
            __syncthreads();
            int nqh = HEADS / NKV;
            if (lane < nqh) {
                int kvh = lane;
                const bf16* kc = kv + (((size_t)layer * NKV + kvh) * 2 + 0) * cap * HDIM;
                const bf16* vc = kv + (((size_t)layer * NKV + kvh) * 2 + 1) * cap * HDIM;
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
#ifdef DEBUG_TINY
                        if (pos == 0 && lane == 0)
                            printf("MK attn h0 t0 score %.6f hi %.6f z %.6f q0 %.6f k0 %.6f v0 %.6f\n",
                                   score, hi, z, qreg[0], tofloat(kc[0]), tofloat(vc[0]));
#endif
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
            __syncthreads();

            // ---- o proj + residual ----
#ifdef DEBUG_TINY
            if (pos == 0 && lane == 0) printf("MK after-attn %.6f %.6f %.6f %.6f\n", sh_attn[0], sh_attn[1], sh_attn[2], sh_attn[3]);
#endif
            for (int i = lane; i < HID; i += 32) sh_o[i] = 0.0f;
            __syncthreads();
            q4<DENSE_I>(QW(oidx), QS(oidx), QB(oidx), sh_attn, sh_o, HID);
            __syncthreads();
            for (int i = lane; i < HID; i += 32) sh_x[i] += sh_o[i];
            __syncthreads();

            // ---- post RMSNorm -> sh_xn ----
#ifdef DEBUG_TINY
            if (pos == 0 && lane == 0) printf("MK after-o-residual %.6f %.6f %.6f %.6f\n", sh_x[0], sh_x[1], sh_x[2], sh_x[3]);
#endif
            rmsnorm(sh_x, B(post), sh_xn);
            __syncthreads();

            // ---- MLP / MoE ----
#ifdef DEBUG_TINY
            if (pos == 0 && lane == 0) printf("MK after-postnorm %.6f %.6f %.6f %.6f\n", sh_xn[0], sh_xn[1], sh_xn[2], sh_xn[3]);
#endif
            if (layer == 0) {
                for (int i = lane; i < DENSE_I; i += 32) { sh_mid[i] = 0.0f; sh_up[i] = 0.0f; }
                for (int i = lane; i < HID; i += 32) sh_o[i] = 0.0f;
                for (int i = lane; i < HID; i += 32) sh_mlp[i] = 0.0f;
                __syncthreads();
                q4<HID, 7>(QW(GATE_L0), QS(GATE_L0), QB(GATE_L0), sh_xn, sh_mid, DENSE_I);
                __syncthreads();
#ifdef DEBUG_TINY
                if (pos == 0 && lane == 0) printf("MK gate-out %.6f %.6f\n", sh_mid[0], sh_mid[1]);
#endif
                q4<HID, 8>(QW(UP_L0), QS(UP_L0), QB(UP_L0), sh_xn, sh_up, DENSE_I);
                __syncthreads();
#ifdef DEBUG_TINY
                if (pos == 0 && lane == 0) printf("MK up-out %.6f %.6f\n", sh_up[0], sh_up[1]);
#endif
                for (int i = lane; i < DENSE_I; i += 32)
                    sh_mid[i] = tobf(tofloat(sh_mid[i]) * tofloat(sh_up[i]));
                __syncthreads();
                q4<DENSE_I>(QW(DOWN_L0), QS(DOWN_L0), QB(DOWN_L0), sh_mid, sh_mlp, HID);
                __syncthreads();
#ifdef DEBUG_TINY
                if (pos == 0 && layer == 0 && lane == 0)
                    printf("MK l0mlp mlp %.6f %.6f mid %.6f %.6f\n", sh_mlp[0], sh_mlp[1], sh_mid[0], sh_mid[1]);
#endif
                for (int i = lane; i < HID; i += 32) sh_x[i] += sh_mlp[i];
                __syncthreads();
#ifdef DEBUG_TINY
                if (pos == 0 && layer == 0 && lane == 0)
                    printf("MK l0mlp-after x %.6f %.6f\n", sh_x[0], sh_x[1]);
#endif
            } else {
                if (lane == 0) {
                    int top[TOPK]; float tv[TOPK];
                    for (int t = 0; t < TOPK; ++t) { tv[t] = -1e30f; top[t] = 0; }
                    const bf16* rw = B(IDX_ROUTER_W);
                    for (int e = 0; e < NEXP; ++e) {
                        float sc = 0.0f;
                        for (int i = 0; i < HID; ++i) sc += tofloat(rw[(size_t)e * HID + i]) * sh_xn[i];
                        sc = tofloat(tobf(sc)) + tofloat(B(IDX_ROUTER_B)[e]);
                        float sig = 1.0f / (1.0f + expf(-sc));
                        sh_score[e] = sig;
                        for (int t = 0; t < TOPK; ++t)
                            if (sig > tv[t]) {
                                for (int u = TOPK - 1; u > t; --u) { tv[u] = tv[u - 1]; top[u] = top[u - 1]; }
                                tv[t] = sig; top[t] = e;
                                break;
                            }
                    }
                    for (int t = 0; t < TOPK; ++t) sh_sel[t] = top[t];
#ifdef DEBUG_TINY
                    if (pos == 0)
                        printf("MK router top %d %d tv %.6f %.6f\n", top[0], top[1], tv[0], tv[1]);
#endif
                }
                __syncthreads();
                if (lane == 0) {
                    float s0 = sh_score[sh_sel[0]], s1 = sh_score[sh_sel[1]];
                    float wsum = s0 + s1 + 1e-20f;
                    sh_wsel[0] = s0 / wsum; sh_wsel[1] = s1 / wsum;
                }
                __syncthreads();
                for (int i = lane; i < HID; i += 32) sh_mlp[i] = 0.0f;
                __syncthreads();
#ifdef DEBUG_TINY
                if (pos == 0 && lane == 0)
                    printf("MK moeptr off25 %ld wptr %ld sptr %ld\n",
                           (long)OFF(Q4_FIRST + SW_GATE),
                           (long)((const uint8_t*)QW(SW_GATE) - bank),
                           (long)((const uint8_t*)QS(SW_GATE) - bank));
#endif
                for (int t = 0; t < TOPK; ++t) {
                    int e = sh_sel[t];
#ifdef DEBUG_TINY
                    if (pos == 0 && lane == 0)
                        printf("MK tloop t%d e %d sel %d %d\n", t, e, sh_sel[0], sh_sel[1]);
#endif
                    __shared__ float sh_mi[MOE_I], sh_u2[MOE_I], sh_dyn[HID];
                    for (int i = lane; i < MOE_I; i += 32) { sh_mi[i] = 0.0f; sh_u2[i] = 0.0f; }
                    for (int i = lane; i < HID; i += 32) sh_dyn[i] = 0.0f;
                    __syncthreads();
                    // per-expert q4: gate/up rows [MOE_I, HID], down rows [HID, MOE_I]
                    // NB: QS/QB return bf16* — per-expert strides are in ELEMENTS
                    // (byte stride / 2), not bytes (rec-hjr).
                    q4<HID, 9>(QW(SW_GATE) + (size_t)e * (MOE_I * HID / 2),
                            QS(SW_GATE) + (size_t)e * (MOE_I * (HID / 64)),
                            QB(SW_GATE) + (size_t)e * (MOE_I * (HID / 64)),
                            sh_xn, sh_mi, MOE_I);
                    q4<HID>(QW(SW_UP) + (size_t)e * (MOE_I * HID / 2),
                            QS(SW_UP) + (size_t)e * (MOE_I * (HID / 64)),
                            QB(SW_UP) + (size_t)e * (MOE_I * (HID / 64)),
                            sh_xn, sh_u2, MOE_I);
                    __syncthreads();
                    for (int i = lane; i < MOE_I; i += 32)
                        sh_mi[i] = tobf(tofloat(sh_mi[i]) * tofloat(sh_u2[i]));
                    __syncthreads();
#ifdef DEBUG_TINY
                    if (pos == 0 && lane == 0)
                        printf("MK exp gate/up t%d mi %.6f %.6f u2 %.6f %.6f\n", t, sh_mi[0], sh_mi[1], sh_u2[0], sh_u2[1]);
#endif
                    q4<MOE_I>(QW(SW_DOWN) + (size_t)e * (HID * MOE_I / 2),
                              QS(SW_DOWN) + (size_t)e * (HID * (MOE_I / 64)),
                              QB(SW_DOWN) + (size_t)e * (HID * (MOE_I / 64)),
                              sh_mi, sh_dyn, HID);
                    __syncthreads();
#ifdef DEBUG_TINY
                    if (pos == 0 && lane == 0)
                        printf("MK exp down t%d dyn %.6f %.6f mi %.6f %.6f\n", t, sh_dyn[0], sh_dyn[1], sh_mi[0], sh_mi[1]);
#endif
                    for (int i = lane; i < HID; i += 32) sh_mlp[i] += sh_wsel[t] * sh_dyn[i];
                    __syncthreads();
                }
#ifdef DEBUG_TINY
                if (pos == 0 && lane == 0)
                    printf("MK moe wsel %.6f %.6f sh_mlp %.6f %.6f sel %d %d scores %.6f %.6f %.6f %.6f\n",
                           sh_wsel[0], sh_wsel[1], sh_mlp[0], sh_mlp[1],
                           sh_sel[0], sh_sel[1], sh_score[0], sh_score[1], sh_score[2], sh_score[3]);
#endif
                // shared expert
                __shared__ float sh_sg[SH_I], sh_su[SH_I], sh_sd[HID];
                for (int i = lane; i < SH_I; i += 32) { sh_sg[i] = 0.0f; sh_su[i] = 0.0f; }
                for (int i = lane; i < HID; i += 32) sh_sd[i] = 0.0f;
                __syncthreads();
                q4<HID>(QW(SH_GATE), QS(SH_GATE), QB(SH_GATE), sh_xn, sh_sg, SH_I);
                q4<HID>(QW(SH_UP), QS(SH_UP), QB(SH_UP), sh_xn, sh_su, SH_I);
                __syncthreads();
                for (int i = lane; i < SH_I; i += 32) sh_sg[i] = tobf(tofloat(sh_sg[i]) * tofloat(sh_su[i]));
                __syncthreads();
                q4<SH_I>(QW(SH_DOWN), QS(SH_DOWN), QB(SH_DOWN), sh_sg, sh_sd, HID);
                __syncthreads();
                for (int i = lane; i < HID; i += 32) sh_mlp[i] += sh_sd[i];
                __syncthreads();
#ifdef DEBUG_TINY
                if (pos == 0 && lane == 0)
                    printf("MK moe-after sh_sd %.6f %.6f sh_mlp %.6f %.6f\n", sh_sd[0], sh_sd[1], sh_mlp[0], sh_mlp[1]);
#endif
                for (int i = lane; i < HID; i += 32) sh_x[i] += sh_mlp[i];
                __syncthreads();
            }
        }

        // ---- final norm + lm head + greedy argmax ----
#ifdef DEBUG_TINY
            if (pos == 0 && lane == 0) printf("MK after-L1-final %.6f %.6f %.6f %.6f\n", sh_x[0], sh_x[1], sh_x[2], sh_x[3]);
#endif
        rmsnorm(sh_x, B(IDX_FNNORM), sh_xn);
        __syncthreads();
        for (int i = lane; i < VOCAB; i += 32) sh_logits[i] = 0.0f;
        __syncthreads();
        q4<HID>(QW(LM_HEAD), QS(LM_HEAD), QB(LM_HEAD), sh_xn, sh_logits, VOCAB);
        __syncthreads();
        if (lane == 0) {
            int best = 0;
            for (int i = 1; i < VOCAB; ++i) if (sh_logits[i] > sh_logits[best]) best = i;
            if (out_last_logits) for (int i = 0; i < VOCAB; ++i) out_last_logits[i] = sh_logits[i];
            int slot = pos - prompt_len + 1;
            if (slot >= 0 && slot < n_out) out_tokens[slot] = best;
            sh_next = best;
        }
        __syncthreads();
    }
}

extern "C" void tiny_launch(const uint8_t* h_bank, const int64_t* h_off, bf16* h_kv,
                            const int* h_prompt, int prompt_len, int* h_out_tokens,
                            int n_out, float* h_out_last_logits) {
    int total = prompt_len + n_out;
    size_t bank_sz = (size_t)h_off[31] + q4_total_bytes(17);
    size_t off_sz = 32 * sizeof(int64_t);
    size_t kv_sz = (size_t)NL * NKV * 2 * (size_t)total * HDIM * sizeof(bf16);

    uint8_t* d_bank; int64_t* d_off; bf16* d_kv; int* d_prompt; int* d_out; float* d_logits;
    hipMalloc(&d_bank, bank_sz);
    hipMalloc(&d_off, off_sz);
    hipMalloc(&d_kv, kv_sz);
    hipMalloc(&d_prompt, (size_t)prompt_len * sizeof(int));
    hipMalloc(&d_out, (size_t)n_out * sizeof(int));
    hipMalloc(&d_logits, h_out_last_logits ? (size_t)VOCAB * sizeof(float) : 0);
    hipMemcpy(d_bank, h_bank, bank_sz, hipMemcpyHostToDevice);
    hipMemcpy(d_off, h_off, off_sz, hipMemcpyHostToDevice);
    hipMemcpy(d_kv, h_kv, kv_sz, hipMemcpyHostToDevice);
    hipMemcpy(d_prompt, h_prompt, (size_t)prompt_len * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_out, h_out_tokens, (size_t)n_out * sizeof(int), hipMemcpyHostToDevice);
    if (h_out_last_logits) hipMemcpy(d_logits, h_out_last_logits, VOCAB * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(tiny_generate, dim3(1), dim3(32), 0, 0,
                       d_bank, d_off, d_kv, d_prompt, prompt_len, d_out, n_out, d_logits);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) fprintf(stderr, "tiny_launch sync error: %s\n", hipGetErrorString(err));
    hipMemcpy(h_out_tokens, d_out, (size_t)n_out * sizeof(int), hipMemcpyDeviceToHost);
    if (h_out_last_logits) hipMemcpy(h_out_last_logits, d_logits, VOCAB * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_bank); hipFree(d_off); hipFree(d_kv); hipFree(d_prompt); hipFree(d_out); hipFree(d_logits);
}
