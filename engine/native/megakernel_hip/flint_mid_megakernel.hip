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

// ---- rec-ohs GEPA search knobs: compile-time -D flags, ONE instantiation per build (no template variants,
// per the rec-9t8 I-cache observation). Defaults reproduce 24f167d exactly (verified: default build npz
// byte-identical to 24f167d). Driver: scripts/gepa_kernel_search.py; knob space: KNOBS there.
#ifndef GEPA_GEN_UNROLL        // generic-K pipelined stage loop unroll (1 = 24f167d)
#define GEPA_GEN_UNROLL 1
#endif
#ifndef GEPA_K512_PIPE         // 1: K=512 uses the pipelined stream instead of VCgen single-stage staging
#define GEPA_K512_PIPE 0
#endif
#ifndef GEPA_GEN_PREFETCH_X    // 0: generic path loads x at use instead of one stage ahead (-16 VGPR in flight)
#define GEPA_GEN_PREFETCH_X 1
#endif
#ifndef GEPA_TOPK_PER_WG       // 1: every WG computes the MoE top-K itself (grid barrier -> __syncthreads)
#define GEPA_TOPK_PER_WG 0
#endif
#ifndef GEPA_RESID_FUSE        // 1: O-proj / dense-down residual add fused into the gemv store (-1 barrier/layer)
#define GEPA_RESID_FUSE 0
#endif
// rec-1f4 Step 2 (Track B, RELAXED summation order): K=2048 g64 group-to-lane mapping, per call site.
//   0 = 24f167d strided 4x16 slices (byte-identical reference)
//   1 = one whole g64 group per lane (32 groups/row = 32 lanes), 2x16 B loads + 1 s/b pair per row
//   2 = half-groups: lane loads contiguous 16 B at row+16*lane and row+512+16*lane (fully coalesced), 2 s/b pairs
//   3 = packed row-pair staged through q4_gen_pool (2 KiB/warp, 16 B coalesced loads), lane consumes one
//       whole group from LDS (the "packed-group LDS staging" design)
#ifndef GEPA_S2_MOE            // segmented MoE routed + shared gate/up
#define GEPA_S2_MOE 0
#endif
#ifndef GEPA_S2_QKV            // segmented Q/K/V
#define GEPA_S2_QKV 0
#endif
#ifndef GEPA_S2_DENSE          // non-segmented K=2048 calls via q4_block512 (dense-layer gate/up)
#define GEPA_S2_DENSE 0
#endif
#ifndef GEPA_S2_LMH            // lm_head (150144 x 2048)
#define GEPA_S2_LMH 0
#endif
// rec-ohs direction 1: idle-warp L2 warm-up of the NEXT weight phase during light phases (attention, top-K,
// MoE act, combine). sync_all opens with an agent-scope fence (vmcnt(0)), so loads cannot ride across a barrier;
// only warps with no work in the light phase prefetch, and they consume locally. GEPA_PF = bytes per idle warp
// per site (0 = off, byte-identical). Window = contiguous prefix of the next phase in consumption order.
#ifndef GEPA_PF
#define GEPA_PF 0
#endif
// rec-ohs direction 2: MoE act folded into the down phase. Each workgroup recomputes act (same formula, same
// inputs -> identical values) into sh_xn (dead between gate/up and the next RMSNorm) and the down gemvs read it
// from LDS: one grid barrier per MoE layer becomes a __syncthreads.
#ifndef GEPA_ACT_FOLD
#define GEPA_ACT_FOLD 0
#endif
// rec-ohs candidate 1 (relaxed order): O-proj (K=4096) as two K=2048 halves on the K=2048 fast path (row stride
// 2048 B via Q4Seg::ld = 2), partials in ws_o and ws_down, summed in the residual phase.
#ifndef GEPA_OSPLIT
#define GEPA_OSPLIT 0
#endif
// rec-ohs candidate 2 (relaxed order): MoE down (2 x K=512 routed + K=1024 shared, N=2048) as ONE virtual K=2048 gemv
// over [w0*act0 | w1*act1 | act_sh]; routing weights folded into act, combine becomes ws_x += d.
#ifndef GEPA_DOWNCAT
#define GEPA_DOWNCAT 0
#endif
// rec-ohs run 3 (per-warp depth, summation order IDENTICAL -> byte-identical outputs):
//   GEPA_K512_DUAL: K=512 VCgen path stages TWO quads (8 rows, the full 2 KiB warp pool) per load round, all four
//                   16 B loads issued before any LDS store (2x bytes in flight, half the exposed latencies).
//   GEPA_GEN_DIST2: generic pipelined path (K=1024/4096/6144) prefetches 2 stages ahead (stage t in LDS, t+1 and t+2
//                   in registers); x is read at use (same values) to pay for the extra stage's registers.
#ifndef GEPA_K512_DUAL
#define GEPA_K512_DUAL 0
#endif
#ifndef GEPA_GEN_DIST2
#define GEPA_GEN_DIST2 0
#endif
// rec-fut run 4 (byte-identical):
//   GEPA_ACTX:   MoE act fused into the down gemvs' x-load (x = act(gate, up) computed at load with the act phase's
//                exact formula); the act phase and its grid barrier disappear, no LDS buffer.
//   GEPA_SHRINK: partial-quad tail paths (never taken on mid/full shapes) moved out of line -> smaller hot stream.
#ifndef GEPA_ACTX
#define GEPA_ACTX 0
#endif
// Bedivere M3.2 (RELAXED: softmax merge order): split-K (flash-decoding) attention. When the context length exceeds
// GEPA_ATTN_SPLIT_MIN, each (q-head, position-chunk) gets a warp (chunks = groups / heads), partials (acc[128], max,
// denom) go to the lm_head logits buffer (free during attention), and after one extra grid barrier a merge folds in the
// sink and normalises. At or below the threshold the original one-warp-per-head path runs unchanged.
#ifndef GEPA_ATTN_SPLIT
#define GEPA_ATTN_SPLIT 0
#endif
// rec-e5d iter 2, candidate (2): GQA-shared split-K. Above GEPA_ATTN_SPLIT_MIN, one warp per (kv-head, position chunk)
// computes ALL MID_HEADS/MID_NKV q-heads of that kv-head, so each K/V row is read once per kv-head instead of once per
// q-head (8x less K/V traffic; at depth the per-q-head re-read is DRAM-bound). Partials + sink-aware merge as split-K.
// GEPA_ATTN_LOP (candidate 1, lanes-over-positions, applies to the GEPA_ATTN_SPLIT path): 0 = one position per
// warp-iteration (32-lane dot); 4 = 4 positions per iteration, 8 lanes x 16 dims each (3-step reduction), groups merged
// at chunk end. (Not offered on the GQA path: 8 q-heads x 16 dims x {q, acc} per lane would spill.)
#ifndef GEPA_ATTN_GQA
#define GEPA_ATTN_GQA 0
#endif
#ifndef GEPA_ATTN_GQA_MAXCH
#define GEPA_ATTN_GQA_MAXCH 32
#endif
#ifndef GEPA_ATTN_LOP
#define GEPA_ATTN_LOP 0
#endif
#ifndef GEPA_ATTN_LOP_PF
#define GEPA_ATTN_LOP_PF 0
#endif
#ifndef GEPA_ATTN_LOP_UNROLL
#define GEPA_ATTN_LOP_UNROLL 1
#endif
// rec-e5d iter 5 (charter bandwidth levers on the depth path):
// GEPA_ATTN_NSPLIT: split-K chunks per head (0 = auto: groups / heads, capped at 16). Lever (a).
// GEPA_ATTN_LDS: lever (b) LDS staging. When groups == heads * n_split (one unit per warp; mid and full at 16 workers), workgroup g
//   owns (kv-head g / n_split, chunk g % n_split) and its 8 warps are that kv-head's 8 q-heads. Each K/V tile (32 positions bf16 /
//   16 fp32) is copied to LDS once by the whole workgroup (wide uint4 loads) and read by all 8 heads, so global K/V traffic drops 8x.
//   The tile reuses the gemv pool (q4_gen_db, 16 KiB, idle during attention): no LDS growth. Per-warp position order is identical
//   to the GEPA_ATTN_LOP=8 path, so outputs are expected byte-identical. Requires GEPA_ATTN_LOP == 8.
#ifndef GEPA_ATTN_NSPLIT
#define GEPA_ATTN_NSPLIT 0
#endif
#ifndef GEPA_ATTN_LDS
#define GEPA_ATTN_LDS 0
#endif
// GEPA_ATTN_LASTMERGE: lever (d), b6-class barrier cut on the split path. Each (head, chunk) unit publishes its partial (release
//   fence + agent-scope acq_rel fetch_add on a per-head counter); the unit that arrives last merges that head (same code, same
//   chunk order -> expected byte-identical). Removes the partials->merge grid barrier: one SYNC_ALL per split-K layer.
//   Counters live in the ws_f_logits tail (past every partial layout) and are zeroed in the KV-write phase, behind its barrier.
#ifndef GEPA_ATTN_LASTMERGE
#define GEPA_ATTN_LASTMERGE 0
#endif
#if GEPA_ATTN_LASTMERGE && (!GEPA_ATTN_SPLIT || GEPA_ATTN_GQA)
#error "GEPA_ATTN_LASTMERGE applies to the GEPA_ATTN_SPLIT path"
#endif
#if GEPA_ATTN_LDS && GEPA_ATTN_LOP != 8
#error "GEPA_ATTN_LDS requires GEPA_ATTN_LOP=8"
#endif
#if GEPA_ATTN_GQA && GEPA_ATTN_SPLIT
#error "GEPA_ATTN_GQA and GEPA_ATTN_SPLIT are alternative depth-attention paths"
#endif
#ifndef GEPA_ATTN_SPLIT_MIN
#define GEPA_ATTN_SPLIT_MIN 64
#endif
// rec-3cr race hunt: GEPA_DUMP=1 makes workgroup 0 copy per-(local step, layer) tensors right after the barrier
// that completes them (q after QKV+RoPE, attention out, x after attn residual, router scores, x after MLP) into a
// device buffer set by mid_dump_setup(); repeats are diffed on the host to find the first divergent phase.
// rec-3cr fix candidate: the per-warp LDS staging relies on wavefront-scope fence + wave_barrier pairs, which order
// only atomics (none here) and are not compiler memory barriers for plain LDS ops; GEPA_LDS_SYNC=1 replaces them
// with s_waitcnt lgkmcnt(0) as an asm memory clobber (hardware wait + compiler barrier).
#ifndef GEPA_LDS_SYNC
#define GEPA_LDS_SYNC 0
#endif
#if GEPA_LDS_SYNC
#define WAVE_LDS_PRE()  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory")
#define WAVE_LDS_POST() asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory")
#else
#define WAVE_LDS_PRE()  do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront"); __builtin_amdgcn_wave_barrier(); } while (0)
#define WAVE_LDS_POST() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront"); __builtin_amdgcn_wave_barrier(); \
                             __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront"); } while (0)
#endif
#ifndef GEPA_RACE_EXP
#define GEPA_RACE_EXP 0   // rec-3cr diagnostics only (3: double barrier before O residual, 4: atomic ws_o read)
#endif
#ifndef GEPA_DUMP
#define GEPA_DUMP 0
#endif
#ifndef GEPA_SHRINK
#if defined(FLINT_CFG_FULL) && FLINT_CFG_FULL
// rec-11b: SHRINK=1 is allowed at full config again, but ONLY with the explicit VMEM fence
// (gepa_vmem_fence) at every q4_rows_tail call site. rec-3cr evidence: the out-of-line tail
// codegen raced nondeterministically (O-proj K=4096, row 0 of quads; 3 codegens/3 answers),
// and the race-window disasm shows the caller keeps vmcnt(2) outstanding ACROSS the call and
// the callee entry waits vmcnt(0)/expcnt(0)/lgkmcnt(0) but NOT vscnt — while the caller holds
// 144 B of scratch spill traffic (90 stores / 111 reloads, s1 k_gfx1150.s). The fence closes
// the same-type-WAW/WAR + vscnt-never-waited corner (class B, sunny memo 6dbb2bc) at the
// boundary instead of pinning the codegen.
#define GEPA_SHRINK 1
#else
#define GEPA_SHRINK 1   // PROMOTED (rec-fut run 4, c033): +3.05/+3.59 % ABBA, replicate +2.65/+2.27 %, byte-identical to 24f167d
#endif
#endif
#define GEPA_STR(x) #x
#define GEPA_UNROLL(n) _Pragma(GEPA_STR(unroll n))
#if GEPA_RESID_FUSE
#define GEPA_ACC_DECL , bool gepa_acc = false
#define GEPA_ACC_ARG , gepa_acc
#define GEPA_OUT(p, i, v) ((p)[i] = gepa_acc ? (p)[i] + (v) : (v))
#else
#define GEPA_ACC_DECL
#define GEPA_ACC_ARG
#define GEPA_OUT(p, i, v) ((p)[i] = (v))
#endif

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

// x element for a gemv: XU == 0 -> x[i]; XU > 0 -> MoE act of (gate = x[i], up = x[i + XU]) with the act phase's
// exact formula (XPURE: fp32 path; else the bf16-rounded path).
template <int XU, bool XPURE>
__device__ inline float xval(const float* x, int i) {
    #pragma clang fp contract(off)   // the act phase stores act before any add: never fuse it into the caller's sums
    if constexpr (XU == 0) {
        return x[i];
    } else if constexpr (XPURE) {
        return silu_f(x[i]) * x[i + XU];
    } else {
        bf16 act = tobf(silu_f(tofloat(tobf(x[i]))) * tofloat(tobf(x[i + XU])));
        return tofloat(act);
    }
}
template <int XU, bool XPURE>
__device__ inline float load16_x(const float* x, float* v) {
    #pragma clang fp contract(off)
    if constexpr (XU == 0) {
        return load16_f(x, v);
    } else {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            float x0 = xval<XU, XPURE>(x, i), x1 = xval<XU, XPURE>(x, i + 1);
            float x2 = xval<XU, XPURE>(x, i + 2), x3 = xval<XU, XPURE>(x, i + 3);
            sum += x0 + x1 + x2 + x3;
            v[i]     = x0;
            v[i + 1] = x1 * (1.0f / 16.0f);
            v[i + 2] = x2 * (1.0f / 256.0f);
            v[i + 3] = x3 * (1.0f / 4096.0f);
        }
        return sum;
    }
}

// rec-6ob (option A, explicit contraction): the q4 dot product's floating-point graph is spelled out
// with fmaf and fp-contract is OFF inside, so the compiled arithmetic no longer depends on the fusion
// context the caller is compiled in (under the default -ffp-contract=fast, restructuring a caller's
// loads changed which mul+add pairs fused -> 1e-7 diffs, amplified by bf16-KV into token flips).
// Graph: per 4-nibble group G = fma(x3,q3, fma(x2,q2, fma(x1,q1, x0*q0))); a = ((G0 + G1) + G2) + G3
// (the original left-to-right order); return fma(s, a, sum * b).
// Packed words are passed BY VALUE: reading a register array back through a punned pointer
// (&uint2[] -> uint32_t*) corrupted ~1 in 50k lm_head rows on gfx1150.
__device__ inline float dot16_group(uint32_t u, const float* x) {
    #pragma clang fp contract(off)
    float g = x[0] * (float)(u & 0x0000000fu);
    g = __builtin_fmaf(x[1], (float)(u & 0x000000f0u), g);
    g = __builtin_fmaf(x[2], (float)(u & 0x00000f00u), g);
    g = __builtin_fmaf(x[3], (float)(u & 0x0000f000u), g);
    return g;
}

__device__ inline float dot16v(uint32_t u01, uint32_t u23, const float* x, float s, float b, float sum) {
    #pragma clang fp contract(off)
    float a = dot16_group(u01, x);
    a = a + dot16_group(u01 >> 16, x + 4);
    a = a + dot16_group(u23, x + 8);
    a = a + dot16_group(u23 >> 16, x + 12);
    return __builtin_fmaf(s, a, sum * b);
}

__device__ inline float dot16(const uint8_t* w, const float* x, float s, float b, float sum) {
    const uint32_t* p32 = (const uint32_t*)w;
    return dot16v(p32[0], p32[1], x, s, b, sum);
}

// rec-1f4 Step 2: unscaled nibble dot over 32 contiguous elements (one uint4 BY VALUE, see note above),
// x pre-scaled with the period-4 rule. Left-to-right over the eight 4-nibble groups.
__device__ inline float dot32_raw(uint4 p, const float* x) {
    #pragma clang fp contract(off)
    float a = dot16_group(p.x, x);
    a = a + dot16_group(p.x >> 16, x + 4);
    a = a + dot16_group(p.y, x + 8);
    a = a + dot16_group(p.y >> 16, x + 12);
    a = a + dot16_group(p.z, x + 16);
    a = a + dot16_group(p.z >> 16, x + 20);
    a = a + dot16_group(p.w, x + 24);
    a = a + dot16_group(p.w >> 16, x + 28);
    return a;
}

// x[0..n) -> pre-scaled v[0..n) (load16_f rule), returns the unscaled sum (left-to-right per 16)
template <int NX>
__device__ inline float load_n_f(const float* x, float* v) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < NX; i += 16) sum += load16_f(x + i, v + i);
    return sum;
}

// Unified device function for fused INT4-affine dequant-matmul (rec-aip / rec-tan)
// consuming the zero-copy layout from docs/rocm-model-layer.md s2:
//   w: uint8_t [N, K/2] (2 nibbles/byte, low nibble = even element),
//   s, b: bf16 [N, K/64] (group size 64 along K, stride in ELEMENTS for bf16*).
// Shared by both standalone `flint_q4_gemv_kernel` and persistent `mid_generate_impl`
// (`lm_head` [150144, 2048] and layer Q4 projections).
// For compile-time K == 2048 (`lm_head`, `q_proj`, `k_proj`, `v_proj`, `gate/up`),
// preloads the lane's 4x16 pre-scaled input slices and 4 group sums into registers once
// outside the row loop, eliminating redundant input reads and pre-scale math across rows
// (especially across the 1,173 row-quads per warp in the 150144x2048 lm_head).

// rec-6ob step 4 VC: LDS staging of one 512-element block of a full row-quad. The 32-lane logical warp
// copies 4 rows x 256 B with two 16-byte loads per lane, then each lane reads back exactly the 8 bytes
// it consumes in the unstaged path (same words -> same arithmetic). A logical warp lives inside one
// hardware wave on wave32 and wave64, so wave-level fences order the LDS write -> read -> rewrite.
__device__ inline void q4_stage_block(const uint8_t* __restrict__ w, size_t row_bytes, int first, int byte_off,
                                      int lane, uint8_t* lds /* 1024 B per logical warp */) {
    WAVE_LDS_PRE();  // previous block's reads are done before overwrite
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
        int idx = t * 32 + lane;           // 0..63 : (row 0..3) x (16-byte chunk 0..15)
        int r = idx >> 4, c = idx & 15;
        uint4 q = *(const uint4*)(w + (size_t)(first + r) * row_bytes + byte_off + c * 16);
        *(uint4*)(lds + r * 256 + c * 16) = q;
    }
    WAVE_LDS_POST();
}

// rec-9t8: ONE 2 KiB LDS double buffer per logical warp, shared by every generic-K gemv instantiation.
// (A function-local __shared__ per template instantiation costs 8+ KiB each; rec-6ob.1 showed that
// mid_generate LDS growth past ~33 KiB breaks persistent-workgroup co-residency on dusk.) A warp's gemv
// ops run one after another and every rewrite is behind a wave fence pair, so sharing is safe.
__device__ inline uint8_t* q4_gen_pool(int warp) {
    __shared__ __attribute__((aligned(16))) uint8_t q4_gen_db[8][2048];
    return q4_gen_db[warp];
}

// rec-9t8 lever 3b: K=2048 multi-matrix ("segmented") gemv. Matrices that share the input x (Q/K/V; all
// routed + shared expert gate/up) are one round-robin stream of row-quads over the concatenated quad list,
// with ONE x preload, instead of one call (and one preload + ragged tail) per matrix. Every row's
// arithmetic is unchanged. The table lives in LDS (uniform reads); with no table the quad assignment is
// exactly the single-matrix one (quad gq = sg + k*groups -> first = 4*gq).
// warp `widx` warms bytes [base_off + widx*GEPA_PF, +GEPA_PF) of the concatenation of up to 3 regions.
// Out of line (idle warps only -> keeps it off the hot instruction stream), 4 x 16 B loads in flight per batch.
__device__ __attribute__((noinline)) uint32_t pf_warm_impl(const uint8_t* r0, size_t n0, const uint8_t* r1, size_t n1,
                                                           const uint8_t* r2, size_t n2, size_t lo, int lane) {
    uint32_t acc = 0;
    #pragma unroll 1
    for (int bt = 0; bt < (GEPA_PF > 0 ? GEPA_PF : 512) / 2048; ++bt) {
        uint4 v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            size_t o = lo + (size_t)(bt * 4 + i) * 512 + lane * 16;
            const uint8_t* p = o < n0 ? r0 + o : (o - n0 < n1 ? r1 + (o - n0) : (o - n0 - n1 < n2 ? r2 + (o - n0 - n1) : nullptr));
            v[i] = p ? *(const uint4*)p : make_uint4(0u, 0u, 0u, 0u);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc ^= v[i].x ^ v[i].w;
    }
    return acc;
}
__device__ inline void pf_warm(const uint8_t* r0, size_t n0, const uint8_t* r1, size_t n1, const uint8_t* r2, size_t n2,
                               size_t base_off, int widx, int lane, uint32_t& sink) {
    if constexpr (GEPA_PF > 0) {
        static_assert(GEPA_PF % 2048 == 0, "GEPA_PF must be a multiple of 2048");
        sink ^= pf_warm_impl(r0, n0, r1, n1, r2, n2, base_off + (size_t)widx * GEPA_PF, lane);
    }
}

struct Q4Seg {
    const uint8_t* w;
    const bf16* s;
    const bf16* b;
    float* out;
    int n;
    int ld;       // row stride in K=2048 rows (1024 B packed, 32 s/b); 0 (5-field init) means 1; GEPA_OSPLIT uses 2
};

// rec-ohs candidate 2: virtual K=2048 gemv whose four 512-element lane chunks come from different matrices.
struct Q4Cat {
    const uint8_t* w[4];
    const bf16* s[4];
    const bf16* b[4];
    int rb[4];     // row bytes
    int gpr[4];    // s/b groups per row
    int k0[4];     // element offset of the chunk inside its matrix row
};
__device__ inline void q4_gemv_cat2048(const Q4Cat& c, const float* __restrict__ x, float* __restrict__ out, int n,
                                       int lane, int sg, int groups) {
    float v[4][16], sum[4];
    int bo[4], go[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        sum[j] = load16_f(x + 512 * j + lane * 16, v[j]);
        bo[j] = (c.k0[j] + lane * 16) >> 1;
        go[j] = (c.k0[j] + lane * 16) >> 6;
    }
    for (int first = sg * 4; first < n; first += groups * 4) {
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = first + r;
            if (row < n) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const size_t gi = (size_t)row * c.gpr[j] + go[j];
                    a[r] += dot16(c.w[j] + (size_t)row * c.rb[j] + bo[j], v[j], tofloat(c.s[j][gi]), tofloat(c.b[j][gi]),
                                  sum[j]);
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            float val = simd_sum(a[r]);
            if (lane == 0 && first + r < n) out[first + r] = val;
        }
    }
}

// Partial-quad rows (n % 4 != 0), unstaged, blocks in order: the exact arithmetic of the in-line tails.
// GEPA_SHRINK calls this out of line so the never-taken tails leave the hot instruction stream.
// rec-11b: the boundary fence is FULL-CONFIG ONLY (FLINT_CFG_FULL) — the promoted mid short-ctx
// build (c033) stays byte-identical to 24f167d, and the race was only ever observed on the full
// K=4096 O-proj path. Mid keeps its proven SHRINK=1 codegen untouched.
#if GEPA_SHRINK && defined(FLINT_CFG_FULL)
// rec-11b class-B boundary fence (see the GEPA_SHRINK comment): fully drain VMEM before
// entering / resuming around the out-of-line tail — loads (vmcnt) AND stores (vscnt, the
// gfx10+ counter the caller never waits). Tails are rare partial quads, so the cost is off
// the hot path. Keep in sync with any new q4_rows_tail call site.
__device__ __forceinline__ void gepa_vmem_fence() {
    asm volatile("s_waitcnt vmcnt(0)\n\ts_waitcnt_vscnt null, 0x0" ::: "memory");
}
#ifndef GEPA_FENCE_TAIL
#define GEPA_FENCE_TAIL 1
#endif
#else
#ifndef GEPA_FENCE_TAIL
#define GEPA_FENCE_TAIL 0
#endif
#endif
template <int K, bool ROUND_BF16, int XU, bool XPURE>
__device__ __attribute__((noinline)) void q4_rows_tail(const uint8_t* __restrict__ w, const bf16* __restrict__ s,
                                                       const bf16* __restrict__ b, const float* __restrict__ x,
                                                       bf16* __restrict__ out_bf, float* __restrict__ out_f,
                                                       int row0, int row1, int lane GEPA_ACC_DECL) {
    #pragma unroll 1
    for (int row = row0; row < row1; ++row) {
        float acc = 0.0f;
        #pragma unroll 1
        for (int block = 0; block < K; block += 512) {
            int scalar = block + lane * 16;
            float v[16];
            float sum = load16_x<XU, XPURE>(x + scalar, v);
            int gi = row * (K / 64) + scalar / 64;
            acc += dot16(w + (size_t)row * (K / 2) + scalar / 2, v, tofloat(s[gi]), tofloat(b[gi]), sum);
        }
        float val = simd_sum(acc);
        if (lane == 0) {
            if (out_bf) out_bf[row] = tobf(val);
            if (out_f)  GEPA_OUT(out_f, row, ROUND_BF16 ? tofloat(tobf(val)) : val);
        }
    }
}

#ifndef GEPA_FORCE_INLINE
#define GEPA_FORCE_INLINE 0   // rec-3cr: 1 = always_inline the gemv (no call / no callee LDS access)
#endif
#if GEPA_FORCE_INLINE
#define GEPA_GEMV_ATTR __attribute__((always_inline))
#else
#define GEPA_GEMV_ATTR
#endif
template <int K, bool ROUND_BF16 = false, int GM = 0, int LDT = 1, int XU = 0, bool XPURE = false>
__device__ inline GEPA_GEMV_ATTR void flint_q4_gemv_op(const uint8_t* __restrict__ w,
                                        const bf16* __restrict__ s,
                                        const bf16* __restrict__ b,
                                        const float* __restrict__ x,
                                        bf16* __restrict__ out_bf,
                                        float* __restrict__ out_f,
                                        int n, int lane, int sg, int groups,
                                        const Q4Seg* segs = nullptr, int nseg = 0 GEPA_ACC_DECL) {
    static_assert(K % 512 == 0, "flint_q4_gemv_op requires K % 512 == 0");
    if constexpr (K == 2048 && GM != 0) {
        // rec-1f4 Step 2: g64 groups mapped to lanes (relaxed summation order vs GM=0, same quad stream).
        float xa[32], xb[32];
        int oa, ob, ga, gb;                              // byte offsets in the row, group indices
        if constexpr (GM == 2) { oa = lane * 16; ob = 512 + lane * 16; ga = lane >> 1; gb = 16 + (lane >> 1); }
        else                   { oa = lane * 32; ob = lane * 32 + 16;  ga = lane;      gb = lane; }
        const float suma = load_n_f<32>(x + oa * 2, xa);
        const float sumb = load_n_f<32>(x + ob * 2, xb);
        const int nsegs = segs ? nseg : 1;
        int base = 0, gq = sg;
        uint8_t* const lds = (GM == 3) ? q4_gen_pool(sg % 8) : nullptr;
        for (int i = 0; i < nsegs; ++i) {
            const uint8_t* const W  = segs ? segs[i].w : w;
            const bf16* const    S  = segs ? segs[i].s : s;
            const bf16* const    Bq = segs ? segs[i].b : b;
            float* const         OF = segs ? segs[i].out : out_f;
            bf16* const          OB = segs ? nullptr : out_bf;
            const int            N  = segs ? segs[i].n : n;
            const int nq = (N + 3) >> 2;
            for (; gq < base + nq; gq += groups) {
                const int first = (gq - base) * 4;
                float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                if (GM == 3 && first + 4 <= N) {
                    // two packed row-pairs (2 KiB each, contiguous: rows are 1024 B) through the warp pool
                    #pragma unroll
                    for (int pr = 0; pr < 2; ++pr) {
                        const uint8_t* src = W + (size_t)(first + 2 * pr) * 1024;
                        WAVE_LDS_PRE();
                        uint4 st[4];
                        #pragma unroll
                        for (int t = 0; t < 4; ++t) st[t] = *(const uint4*)(src + (t * 32 + lane) * 16);
                        #pragma unroll
                        for (int t = 0; t < 4; ++t) *(uint4*)(lds + (t * 32 + lane) * 16) = st[t];
                        WAVE_LDS_POST();
                        #pragma unroll
                        for (int rr = 0; rr < 2; ++rr) {
                            const int row = first + 2 * pr + rr;
                            uint4 pa = *(const uint4*)(lds + rr * 1024 + oa);
                            uint4 pb = *(const uint4*)(lds + rr * 1024 + ob);
                            float sv = tofloat(S[(size_t)row * 32 + ga]), bv = tofloat(Bq[(size_t)row * 32 + ga]);
                            a[2 * pr + rr] = __builtin_fmaf(sv, dot32_raw(pa, xa) + dot32_raw(pb, xb), (suma + sumb) * bv);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        const int row = first + r;
                        if (row < N) {
                            const uint8_t* wr = W + (size_t)row * 1024;
                            uint4 pa = *(const uint4*)(wr + oa);
                            uint4 pb = *(const uint4*)(wr + ob);
                            if constexpr (GM == 2) {
                                float sa = tofloat(S[(size_t)row * 32 + ga]), ba = tofloat(Bq[(size_t)row * 32 + ga]);
                                float sb = tofloat(S[(size_t)row * 32 + gb]), bb = tofloat(Bq[(size_t)row * 32 + gb]);
                                a[r] = __builtin_fmaf(sa, dot32_raw(pa, xa), suma * ba) +
                                       __builtin_fmaf(sb, dot32_raw(pb, xb), sumb * bb);
                            } else {
                                float sv = tofloat(S[(size_t)row * 32 + ga]), bv = tofloat(Bq[(size_t)row * 32 + ga]);
                                a[r] = __builtin_fmaf(sv, dot32_raw(pa, xa) + dot32_raw(pb, xb), (suma + sumb) * bv);
                            }
                        }
                    }
                }
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    float val = simd_sum(a[r]);
                    if (lane == 0 && first + r < N) {
                        if (OB) OB[first + r] = tobf(val);
                        if (OF) GEPA_OUT(OF, first + r, ROUND_BF16 ? tofloat(tobf(val)) : val);
                    }
                }
            }
            base += nq;
        }
    } else if constexpr (K == 2048) {
        float v0[16], v1[16], v2[16], v3[16];
        int sc0 = lane * 16;
        int sc1 = 512 + sc0;
        int sc2 = 1024 + sc0;
        int sc3 = 1536 + sc0;
        float sum0 = load16_f(x + sc0, v0);
        float sum1 = load16_f(x + sc1, v1);
        float sum2 = load16_f(x + sc2, v2);
        float sum3 = load16_f(x + sc3, v3);
        int g0 = sc0 >> 6;
        int g1 = sc1 >> 6;
        int g2 = sc2 >> 6;
        int g3 = sc3 >> 6;

        const int nsegs = segs ? nseg : 1;
        int base = 0;          // first global quad index of segment i
        int gq = sg;           // this warp's next global quad index
        for (int i = 0; i < nsegs; ++i) {
            const uint8_t* const W  = segs ? segs[i].w : w;
            const bf16* const    S  = segs ? segs[i].s : s;
            const bf16* const    Bq = segs ? segs[i].b : b;
            float* const         OF = segs ? segs[i].out : out_f;
            bf16* const          OB = segs ? nullptr : out_bf;
            const int            N  = segs ? segs[i].n : n;
            constexpr size_t     LD = LDT;   // compile-time row stride (GEPA_OSPLIT instantiates LDT = 2 for O-proj only)
            const int nq = (N + 3) >> 2;
            for (; gq < base + nq; gq += groups) {
                const int first = (gq - base) * 4;
                float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int row = first + r;
                    if (row < N) {
                        const uint8_t* wr = W + (size_t)row * 1024 * LD;
                        const bf16*    sr = S + (size_t)row * 32 * LD;
                        const bf16*    br = Bq + (size_t)row * 32 * LD;
                        a[r] += dot16(wr + (sc0 >> 1), v0, tofloat(sr[g0]), tofloat(br[g0]), sum0);
                        a[r] += dot16(wr + (sc1 >> 1), v1, tofloat(sr[g1]), tofloat(br[g1]), sum1);
                        a[r] += dot16(wr + (sc2 >> 1), v2, tofloat(sr[g2]), tofloat(br[g2]), sum2);
                        a[r] += dot16(wr + (sc3 >> 1), v3, tofloat(sr[g3]), tofloat(br[g3]), sum3);
                    }
                }
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    float val = simd_sum(a[r]);
                    if (lane == 0 && first + r < N) {
                        if (OB) OB[first + r] = tobf(val);
                        if (OF) GEPA_OUT(OF, first + r, ROUND_BF16 ? tofloat(tobf(val)) : val);
                    }
                }
            }
            base += nq;
        }
    } else if constexpr (K == 512 && !GEPA_K512_PIPE) {
        // One 512-element block per row: a stage is a whole quad, so pipelining buys only a prologue
        // (measured 0.91x standalone, rec-9t8 b3). Keep the VCgen single-stage staging (39629e7).
        uint8_t* const lds = q4_gen_pool(sg % 8);
#if GEPA_K512_DUAL
        {
            const int scalar = lane * 16;
            float v[16];
            const float sum = load16_x<XU, XPURE>(x + scalar, v);
            int first = sg * 8;
            for (; first + 8 <= n; first += groups * 8) {
                WAVE_LDS_PRE();            // previous round's LDS reads are done
                uint4 q[4];
                #pragma unroll
                for (int t = 0; t < 4; ++t) {               // 128 x 16 B chunks: (row 0..7) x (chunk 0..15)
                    int idx = t * 32 + lane, r = idx >> 4, c = idx & 15;
                    q[t] = *(const uint4*)(w + (size_t)(first + r) * (K / 2) + c * 16);
                }
                float sc[8], bi[8];
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    int gi = (first + r) * (K / 64) + scalar / 64;
                    sc[r] = tofloat(s[gi]);
                    bi[r] = tofloat(b[gi]);
                }
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int idx = t * 32 + lane, r = idx >> 4, c = idx & 15;
                    *(uint4*)(lds + r * 256 + c * 16) = q[t];
                }
                WAVE_LDS_POST();
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const uint32_t* p = (const uint32_t*)(lds + r * 256 + lane * 8);
                    float val = simd_sum(0.0f + dot16v(p[0], p[1], v, sc[r], bi[r], sum));
                    if (lane == 0) {
                        if (out_bf) out_bf[first + r] = tobf(val);
                        if (out_f)  GEPA_OUT(out_f, first + r, ROUND_BF16 ? tofloat(tobf(val)) : val);
                    }
                }
            }
            // remainder rows (n % 8): per row, unstaged, identical arithmetic
            for (int row = first; row < n && row < first + 8; ++row) {
                int gi = row * (K / 64) + scalar / 64;
                float val = simd_sum(0.0f + dot16(w + (size_t)row * (K / 2) + scalar / 2, v, tofloat(s[gi]), tofloat(b[gi]), sum));
                if (lane == 0) {
                    if (out_bf) out_bf[row] = tobf(val);
                    if (out_f)  GEPA_OUT(out_f, row, ROUND_BF16 ? tofloat(tobf(val)) : val);
                }
            }
        }
        if (false)
#endif
        float hv[16], hsum = 0.0f;               // GEPA_ACTX: the lane's x slice is quad-invariant -> act once
        if constexpr (XU != 0) hsum = load16_x<XU, XPURE>(x + lane * 16, hv);
        for (int first = sg * 4; first < n; first += groups * 4) {
            float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int scalar = lane * 16;
            float v[16];
            float sum;
            if constexpr (XU != 0) {
                sum = hsum;
                #pragma unroll
                for (int i = 0; i < 16; ++i) v[i] = hv[i];
            } else {
                sum = load16_f(x + scalar, v);
            }
#if GEPA_SHRINK
            if (first + 4 > n) {
#if GEPA_FENCE_TAIL
                gepa_vmem_fence();   // rec-11b: full config only (GEPA_FENCE_TAIL); mid keeps the c033 tail call unfenced
#endif
                q4_rows_tail<K, ROUND_BF16, XU, XPURE>(w, s, b, x, out_bf, out_f, first, n, lane GEPA_ACC_ARG);
                continue;
            }
#endif
            if (first + 4 <= n) {
                q4_stage_block(w, (size_t)(K / 2), first, 0, lane, lds);
                float sc[4], bi[4];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int gi = (first + r) * (K / 64) + scalar / 64;
                    sc[r] = tofloat(s[gi]);
                    bi[r] = tofloat(b[gi]);
                }
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const uint32_t* p = (const uint32_t*)(lds + r * 256 + lane * 8);
                    a[r] += dot16v(p[0], p[1], v, sc[r], bi[r], sum);
                }
            } else {
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
                    if (out_f)  GEPA_OUT(out_f, first + r, ROUND_BF16 ? tofloat(tobf(val)) : val);
                }
            }
        }
    } else {
        // rec-9t8 lever 2: software-pipelined generic path (K = 1024, 4096, full-config 6144). Full row-quads form one stream of stages
        // (quad, 512-element block). Stage t+1's weights (two b128 per lane), its 4+4 bf16 scales/biases
        // and its 16 raw x floats are loaded into registers before stage t's compute. The weights land
        // in the other half of a per-warp 2 KiB LDS double buffer behind a wave fence pair that guards only
        // that half. Per lane: the same weight words (LDS readback of its own 8 B), the same scales, the same
        // pre-scale arithmetic as load16_f, and the same block order per row -> byte-identical to the
        // VCgen staging path (39629e7) by construction. One LDS pool is shared by every generic K.
        constexpr int NB = K / 512;
        constexpr int G = K / 64;
        uint8_t* const pool = q4_gen_pool(sg % 8);
        const int r_lo = lane >> 4, c16 = (lane & 15) * 16;
        const int step = groups * 4;
        int first = sg * 4;
        uint4 cw0, cw1;            // staged-for-next weights (in flight)
        uint16_t cs[4], cb[4];     // current stage's scale/bias bits
        float cx[16];              // current stage's raw x
        auto load_stage = [&](int f, int j, uint4& w0, uint4& w1, uint16_t* ss, uint16_t* bb, float* xx) {
            const uint8_t* wq = w + (size_t)f * (K / 2) + j * 256;
            w0 = *(const uint4*)(wq + (size_t)r_lo * (K / 2) + c16);
            w1 = *(const uint4*)(wq + (size_t)(r_lo + 2) * (K / 2) + c16);
            const int gl = (j * 512 + lane * 16) / 64;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                ss[r] = ((const uint16_t*)s)[(f + r) * G + gl];
                bb[r] = ((const uint16_t*)b)[(f + r) * G + gl];
            }
            if constexpr (GEPA_GEN_PREFETCH_X && XU != 0) {
                #pragma unroll
                for (int i = 0; i < 16; ++i) xx[i] = xval<XU, XPURE>(x + j * 512 + lane * 16, i);
            } else if constexpr (GEPA_GEN_PREFETCH_X) {
                const float4* xp = (const float4*)(x + j * 512 + lane * 16);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float4 t4 = xp[i];
                    xx[4 * i] = t4.x; xx[4 * i + 1] = t4.y; xx[4 * i + 2] = t4.z; xx[4 * i + 3] = t4.w;
                }
            }
        };
        auto put_stage = [&](uint8_t* buf, const uint4& w0, const uint4& w1) {
            WAVE_LDS_PRE();   // last reader of this half (stage t-1) is done
            *(uint4*)(buf + r_lo * 256 + c16) = w0;
            *(uint4*)(buf + (r_lo + 2) * 256 + c16) = w1;
            WAVE_LDS_POST();
        };
        int par = 0;
#if GEPA_GEN_DIST2
        // linear stage stream (quad f, block j); stage t in LDS, t+1 (r1) and t+2 (r2) in registers.
        auto ld_wsb = [&](int f, int j, uint4& w0, uint4& w1, uint16_t* ss, uint16_t* bb) {
            const uint8_t* wq = w + (size_t)f * (K / 2) + j * 256;
            w0 = *(const uint4*)(wq + (size_t)r_lo * (K / 2) + c16);
            w1 = *(const uint4*)(wq + (size_t)(r_lo + 2) * (K / 2) + c16);
            const int gl = (j * 512 + lane * 16) / 64;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                ss[r] = ((const uint16_t*)s)[(f + r) * G + gl];
                bb[r] = ((const uint16_t*)b)[(f + r) * G + gl];
            }
        };
        if (first + 4 <= n) {
            int tf = first, tj = 0;
            ld_wsb(tf, 0, cw0, cw1, cs, cb);
            put_stage(pool, cw0, cw1);
            int f1 = tf, j1 = 1;
            if (j1 == NB) { f1 += step; j1 = 0; }
            bool v1 = f1 + 4 <= n;
            uint4 r1w0, r1w1;
            uint16_t r1s[4], r1b[4];
            if (v1) ld_wsb(f1, j1, r1w0, r1w1, r1s, r1b);
            float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            while (true) {
                int f2 = f1, j2 = j1 + 1;
                if (j2 == NB) { f2 += step; j2 = 0; }
                const bool v2 = v1 && f2 + 4 <= n;
                uint4 r2w0, r2w1;
                uint16_t r2s[4], r2b[4];
                if (v2) ld_wsb(f2, j2, r2w0, r2w1, r2s, r2b);
                // compute stage t (x at use; pre-scale exactly as load16_f)
                float v[16];
                const float sum = load16_x<XU, XPURE>(x + tj * 512 + lane * 16, v);
                const uint8_t* cur = pool + par * 1024;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const uint32_t* p = (const uint32_t*)(cur + r * 256 + lane * 8);
                    float sc = __uint_as_float((uint32_t)cs[r] << 16);
                    float bi = __uint_as_float((uint32_t)cb[r] << 16);
                    a[r] += dot16v(p[0], p[1], v, sc, bi, sum);
                }
                if (tj == NB - 1) {
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        float val = simd_sum(a[r]);
                        if (lane == 0) {
                            if (out_bf) out_bf[tf + r] = tobf(val);
                            if (out_f)  GEPA_OUT(out_f, tf + r, ROUND_BF16 ? tofloat(tobf(val)) : val);
                        }
                        a[r] = 0.0f;
                    }
                }
                if (!v1) break;
                put_stage(pool + (par ^ 1) * 1024, r1w0, r1w1);
                #pragma unroll
                for (int r = 0; r < 4; ++r) { cs[r] = r1s[r]; cb[r] = r1b[r]; }
                par ^= 1;
                tf = f1; tj = j1;
                f1 = f2; j1 = j2; v1 = v2;
                r1w0 = r2w0; r1w1 = r2w1;
                #pragma unroll
                for (int r = 0; r < 4; ++r) { r1s[r] = r2s[r]; r1b[r] = r2b[r]; }
            }
            while (first + 4 <= n) first += step;   // hand the partial trailing quad (if any) to the tail loop
        }
        if (false)
#else
        if (first + 4 <= n) {
            load_stage(first, 0, cw0, cw1, cs, cb, cx);
            put_stage(pool, cw0, cw1);
        }
#endif
        for (; first + 4 <= n; first += step) {
            const int nfirst = first + step;
            const bool nfull = nfirst + 4 <= n;
            float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            GEPA_UNROLL(GEPA_GEN_UNROLL)
            for (int j = 0; j < NB; ++j) {
                const bool last = j + 1 == NB;
                const bool pf = !last || nfull;
                uint16_t ns[4], nb[4];
                float nx[16];
                if (pf) load_stage(last ? nfirst : first, last ? 0 : j + 1, cw0, cw1, ns, nb, nx);
                if constexpr (!GEPA_GEN_PREFETCH_X && XU != 0) {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) cx[i] = xval<XU, XPURE>(x + j * 512 + lane * 16, i);
                } else if constexpr (!GEPA_GEN_PREFETCH_X) {   // x at use (same values as the prefetched copy)
                    const float4* xp = (const float4*)(x + j * 512 + lane * 16);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float4 t4 = xp[i];
                        cx[4 * i] = t4.x; cx[4 * i + 1] = t4.y; cx[4 * i + 2] = t4.z; cx[4 * i + 3] = t4.w;
                    }
                }
                // compute stage t: pre-scale exactly as load16_f
                float v[16];
                float sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; i += 4) {
                    float x0 = cx[i], x1 = cx[i + 1], x2 = cx[i + 2], x3 = cx[i + 3];
                    sum += x0 + x1 + x2 + x3;
                    v[i]     = x0;
                    v[i + 1] = x1 * (1.0f / 16.0f);
                    v[i + 2] = x2 * (1.0f / 256.0f);
                    v[i + 3] = x3 * (1.0f / 4096.0f);
                }
                const uint8_t* cur = pool + par * 1024;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const uint32_t* p = (const uint32_t*)(cur + r * 256 + lane * 8);
                    float sc = __uint_as_float((uint32_t)cs[r] << 16);
                    float bi = __uint_as_float((uint32_t)cb[r] << 16);
                    a[r] += dot16v(p[0], p[1], v, sc, bi, sum);
                }
                if (pf) {
                    put_stage(pool + (par ^ 1) * 1024, cw0, cw1);
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) { cs[r] = ns[r]; cb[r] = nb[r]; }
                    if constexpr (GEPA_GEN_PREFETCH_X) {
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) cx[i] = nx[i];
                    }
                }
                par ^= 1;
            }
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                float val = simd_sum(a[r]);
                if (lane == 0) {
                    if (out_bf) out_bf[first + r] = tobf(val);
                    if (out_f)  GEPA_OUT(out_f, first + r, ROUND_BF16 ? tofloat(tobf(val)) : val);
                }
            }
        }
        // partial trailing quad (n % 4 != 0): unstaged, one row at a time, blocks in order (as before)
        for (; first < n; first += step) {
#if GEPA_SHRINK
#if GEPA_FENCE_TAIL
            gepa_vmem_fence();
#endif
            q4_rows_tail<K, ROUND_BF16, XU, XPURE>(w, s, b, x, out_bf, out_f, first, first + 4 < n ? first + 4 : n, lane
                                                  GEPA_ACC_ARG);
            continue;
#endif
            #pragma unroll 1
            for (int row = first; row < first + 4 && row < n; ++row) {
                float acc = 0.0f;
                #pragma unroll 1
                for (int block = 0; block < K; block += 512) {
                    int scalar = block + lane * 16;
                    float v[16];
                    float sum = load16_x<XU, XPURE>(x + scalar, v);
                    int gi = row * (K / 64) + scalar / 64;
                    acc += dot16(w + (size_t)row * (K / 2) + scalar / 2, v, tofloat(s[gi]), tofloat(b[gi]), sum);
                }
                float val = simd_sum(acc);
                if (lane == 0) {
                    if (out_bf) out_bf[row] = tobf(val);
                    if (out_f)  GEPA_OUT(out_f, row, ROUND_BF16 ? tofloat(tobf(val)) : val);
                }
            }
        }
    }
}

template <int K, bool ROUND_BF16 = false>
__device__ inline void q4_block512(const uint8_t* w, const bf16* s, const bf16* b,
                                   const float* x, bf16* out_bf, float* out_f,
                                   int n, int lane, int sg, int groups GEPA_ACC_DECL) {
    flint_q4_gemv_op<K, ROUND_BF16, (K == 2048 ? GEPA_S2_DENSE : 0)>(w, s, b, x, out_bf, out_f, n, lane, sg, groups,
                                                                     nullptr, 0 GEPA_ACC_ARG);
}

// ---- Model config (rec-u9q): FLINT_CFG_FULL=0 -> mid-8L (default, run-of-record d1f6208),
//      FLINT_CFG_FULL=1 -> full Flint (RecursiveSI/flint-int4 config.json). Everything that differs
//      between the two is here; the per-token math is shared. ----
#ifndef FLINT_CFG_FULL
#define FLINT_CFG_FULL 0
#endif
#if FLINT_CFG_FULL
constexpr int MID_NL = 49;
constexpr int MID_N_DENSE = 2;          // mlp_only_layers [0, 1] (must be leading)
constexpr int MID_DENSE_I = 6144;       // intermediate_size
constexpr int MID_NEXP = 256;
constexpr int MID_TOPK = 6;
// layer_types full_attention == no_rope_layer_indices: [0,1,5,9,...,45,48]
constexpr uint64_t MID_FULL_MASK = (1ull << 0) | (1ull << 1) | (1ull << 5) | (1ull << 9) | (1ull << 13) |
                                   (1ull << 17) | (1ull << 21) | (1ull << 25) | (1ull << 29) | (1ull << 33) |
                                   (1ull << 37) | (1ull << 41) | (1ull << 45) | (1ull << 48);
#else
constexpr int MID_NL = 8;
constexpr int MID_N_DENSE = 1;          // mlp_only_layers [0]
constexpr int MID_DENSE_I = 2048;
constexpr int MID_NEXP = 32;
constexpr int MID_TOPK = 2;
constexpr uint64_t MID_FULL_MASK = 0xAAull; // odd layers full attention (no rope)
#endif
static_assert(MID_NL <= 64, "MID_FULL_MASK is 64-bit");
static_assert(MID_TOPK <= 16, "ws_i_sel has 16 slots");
static_assert(MID_TOPK <= 32, "ws_f_wsel has 32 slots");
constexpr int MID_N_FULL = __builtin_popcountll(MID_FULL_MASK);
constexpr int MID_N_SLIDING = MID_NL - MID_N_FULL;
__host__ __device__ constexpr bool mid_layer_full(int l) { return (MID_FULL_MASK >> l) & 1ull; }
// index of layer l among the full (resp. sliding) layers, for the per-type KV caches
__host__ __device__ constexpr int mid_full_idx(int l) { return __builtin_popcountll(MID_FULL_MASK & ((1ull << l) - 1ull)); }
__host__ __device__ constexpr int mid_sliding_idx(int l) { return l - mid_full_idx(l); }
constexpr int MID_HID = 2048;
constexpr int MID_HEADS = 32;
constexpr int MID_NKV = 4;
constexpr int MID_HDIM = 128;
constexpr int MID_QDIM = MID_HEADS * MID_HDIM;   // 4096
constexpr int MID_KVDIM = MID_NKV * MID_HDIM;    // 512
constexpr int MID_MOE_I = 512;
constexpr int MID_SH_I = 1024;
constexpr int MID_VOCAB = 150144;
constexpr int MID_WIN = 513;
constexpr float MID_EPS = 1e-6f;
constexpr float MID_ATTN_SCALE = 0.08838834764831845f; // 1/sqrt(128)
constexpr float MID_ROPE_LOGBASE = 13.28771237954945f; // log2(10000)
constexpr int mid_max3(int a, int b, int c) { return a > b ? (a > c ? a : c) : (b > c ? b : c); }
constexpr int MID_FFN_MAX = mid_max3(MID_DENSE_I, MID_SH_I, MID_MOE_I);

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
// Generalized: 7*NL dense entries, final norm, embed, 10*NL q4 entries, lm_head, size sentinel
// (mid-8L: 56/57/58/138/140; full Flint NL=49: 343/344/345/835/837 = pack_flint_bank.py).
constexpr int MID_OFF_FINAL_NORM = 7 * MID_NL;
constexpr int MID_OFF_EMBED = 7 * MID_NL + 1;
constexpr int MID_OFF_LAYER_Q4_BASE = 7 * MID_NL + 2;
constexpr int MID_OFF_LM_HEAD = MID_OFF_LAYER_Q4_BASE + 10 * MID_NL;
constexpr int MID_NUM_OFFSETS = MID_OFF_LM_HEAD + 2;
static_assert(FLINT_CFG_FULL || (MID_OFF_LM_HEAD == 138 && MID_NUM_OFFSETS == 140), "mid layout drifted");
static_assert(!FLINT_CFG_FULL || MID_NUM_OFFSETS == 837, "full layout must match pack_flint_bank (837)");

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

constexpr int MID_WS_SCORES = MID_NEXP > 32 ? MID_NEXP : 32;
constexpr int MID_WS_LOGITS_OFF = 2 * MID_HID + 2 * MID_QDIM + 2 * MID_KVDIM + 3 * MID_FFN_MAX +
                                  (MID_TOPK + 1) * MID_HID + MID_WS_SCORES + 32 + 64;
static_assert(FLINT_CFG_FULL || MID_WS_LOGITS_OFF == 25728, "mid workspace layout drifted");

constexpr int MID_DUMP_Q = 0, MID_DUMP_ATTN = MID_QDIM, MID_DUMP_XA = 2 * MID_QDIM, MID_DUMP_SC = 2 * MID_QDIM + MID_HID,
              MID_DUMP_XM = 2 * MID_QDIM + MID_HID + MID_WS_SCORES, MID_DUMP_O = 2 * MID_QDIM + 2 * MID_HID + MID_WS_SCORES,
              MID_DUMP_STRIDE = 2 * MID_QDIM + 3 * MID_HID + MID_WS_SCORES;
__device__ float* g_mid_dump = nullptr;
__device__ int g_mid_dump_steps = 0;
#if GEPA_DUMP
#define MID_DUMP(slot, src, n)                                                                                  \
    if (group == 0 && g_mid_dump != nullptr && lp < g_mid_dump_steps) {                                         \
        float* d_ = g_mid_dump + ((size_t)lp * MID_NL + layer) * MID_DUMP_STRIDE + (slot);                      \
        for (int i_ = tid; i_ < (n); i_ += blockDim.x) d_[i_] = (src)[i_];                                       \
    }
#else
#define MID_DUMP(slot, src, n)
#endif

template <typename KVT, bool PURE_FP32>
__device__ inline void mid_generate_impl(
    const uint8_t* bank, const int64_t* off,
    KVT* kv_sliding,                         // [N_SLIDING][2 (K|V)][4][513][128] (mid-8L: layers 0,2,4,6)
    KVT* kv_full,                            // [N_FULL][2 (K|V)][4][full_cap][128] (mid-8L: layers 1,3,5,7)
    int full_cap,
    const int* prompt, int prompt_len,
    int* out_tokens, int n_out,
    float* out_step_logits, float* out_last_logits,
    bf16* ws_bf, float* ws_f, int* ws_i,
    uint32_t* control, int workers,
    int* out_chosen_experts = nullptr,
    int start_pos = 0) {   // Bedivere M3.1: absolute position of local step 0 (KV for [0, start_pos) already in the session)

    const int tid = threadIdx.x;          // 0..255
    const int lane = tid & 31;            // 0..31
    const int group = blockIdx.x;         // 0..workers-1
    const int sg = group * 8 + (tid >> 5);// 0..workers*8-1
    const int groups = workers * 8;
    const int gtid = group * 256 + tid;
    const int gthreads = workers * 256;
    const int total = prompt_len + n_out;

    // Partition device workspace pointers (float precision matching tiny_megakernel.hip)
    // Sequential layout (mid-8L offsets unchanged: act 23552, scores 25600, logits 25728).
    float* ws_x      = ws_f + 0;
    float* ws_q      = ws_x + MID_HID;
    float* ws_k      = ws_q + MID_QDIM;
    float* ws_v      = ws_k + MID_KVDIM;
    float* ws_attn   = ws_v + MID_KVDIM;
    float* ws_o      = ws_attn + MID_QDIM;
    float* ws_gate   = ws_o + MID_HID;
    float* ws_up     = ws_gate + MID_FFN_MAX;
    float* ws_down   = ws_up + MID_FFN_MAX;           // (TOPK routed + 1 shared) * HID
    float* ws_f_act    = ws_down + (MID_TOPK + 1) * MID_HID;
    float* ws_f_scores = ws_f_act + MID_FFN_MAX;      // NEXP slots (>= 32)
    float* ws_f_wsel   = ws_f_scores + MID_WS_SCORES;  // 32 slots
    float* ws_f_maxval = ws_f_wsel + 32;              // 64 slots
    float* ws_f_logits = ws_f_maxval + 64;
    (void)ws_bf;

    int* ws_i_sel    = ws_i + 0;
    int* ws_i_maxidx = ws_i + 16;

    __shared__ float sh_xn[MID_HID];
    __shared__ Q4Seg sh_seg[2 * MID_TOPK + 2];   // rec-9t8 3b segment table (QKV: 3, MoE gate/up: 2*TOPK+2)
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

    uint32_t pf_sink = 0;   // rec-ohs direction 1 (consumed after the loop under an impossible-at-runtime guard)
    for (int lp = 0; lp < total; ++lp) {                 // lp: local step (prompt/out/logit indexing)
        const int pos = start_pos + lp;                  // pos: absolute position (RoPE, KV slot, attention range)
        int token = (lp < prompt_len) ? prompt[lp] : out_tokens[lp - prompt_len];

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
            int is_sliding = !mid_layer_full(layer);
            int rotary = is_sliding; // Flint: no_rope_layer_indices == full-attention layers (both configs)

            // ---- 2. Input RMSNorm (replicated in each workgroup's sh_xn) ----
            mid_rmsnorm_2048(ws_x, ln_w, sh_xn, sh_red);

            // ---- 3. Q, K, V projections (K=2048) ----
            // rec-9t8 3b: Q, K, V as one segmented K=2048 gemv (one x preload, one quad stream)
            if (tid == 0) {
                sh_seg[0] = {Q_W(qbase + 0), Q_S(qbase + 0, MID_QDIM, MID_HID), Q_B(qbase + 0, MID_QDIM, MID_HID), ws_q, MID_QDIM};
                sh_seg[1] = {Q_W(qbase + 1), Q_S(qbase + 1, MID_KVDIM, MID_HID), Q_B(qbase + 1, MID_KVDIM, MID_HID), ws_k, MID_KVDIM};
                sh_seg[2] = {Q_W(qbase + 2), Q_S(qbase + 2, MID_KVDIM, MID_HID), Q_B(qbase + 2, MID_KVDIM, MID_HID), ws_v, MID_KVDIM};
            }
            __syncthreads();
            flint_q4_gemv_op<MID_HID, false, GEPA_S2_QKV>(nullptr, nullptr, nullptr, sh_xn, nullptr, nullptr, 0, lane, sg, groups, sh_seg, 3);
            SYNC_ALL();

            // ---- 4. Per-head RMSNorm + NeoX RoPE on ws_q (32 heads) and ws_k (4 heads) ----
            mid_head_norm_rope(ws_q, qn_w, MID_HEADS, rotary, pos, lane, sg, groups);
            mid_head_norm_rope(ws_k, kn_w, MID_NKV, rotary, pos, lane, sg, groups);
            SYNC_ALL();

            MID_DUMP(MID_DUMP_Q, ws_q, MID_QDIM);
            // ---- 5. Write K, V to head-major KV cache (KVT = bf16 or float) ----
            int l_idx = is_sliding ? mid_sliding_idx(layer) : mid_full_idx(layer); // mid-8L: layer / 2
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
#if GEPA_ATTN_LASTMERGE
            if (sg == 0 && lane < MID_HEADS)                   // per-head arrival counters for this layer's split merge
                reinterpret_cast<uint32_t*>(ws_f_logits + MID_VOCAB - 64)[lane] = 0u;
#endif
            SYNC_ALL();

            // ---- 6. Attention with Sinks (32 warps across workgroups, 1 q-head per warp) ----
            int t_start = is_sliding ? ((pos - (MID_WIN - 1) > 0) ? (pos - (MID_WIN - 1)) : 0) : 0;
#if GEPA_ATTN_GQA
            const int n_ctx = pos - t_start + 1;
            constexpr int GQ = MID_HEADS / MID_NKV;              // q-heads per kv-head (8)
            const int n_ch = groups / MID_NKV < GEPA_ATTN_GQA_MAXCH ? groups / MID_NKV : GEPA_ATTN_GQA_MAXCH;
            if (n_ctx > GEPA_ATTN_SPLIT_MIN && n_ch > 1) {       // uniform across workgroups -> extra barrier is safe
                constexpr int PSTR = MID_HDIM + 2;                // per partial: acc[128], max, denom
                static_assert(MID_HEADS * GEPA_ATTN_GQA_MAXCH * PSTR <= MID_VOCAB, "GQA partials overflow ws_f_logits");
                float* const part = ws_f_logits;                  // [heads][n_ch][PSTR], free until lm_head
                const int clen = (n_ctx + n_ch - 1) / n_ch;
                for (int u = sg; u < MID_NKV * n_ch; u += groups) {
                    const int kvh = u / n_ch, c = u % n_ch;
                    const KVT* kc = l_kv + (0 * MID_NKV + (size_t)kvh) * l_cap * MID_HDIM;
                    const KVT* vc = l_kv + (1 * MID_NKV + (size_t)kvh) * l_cap * MID_HDIM;
                    const int ta = t_start + c * clen;
                    const int tb = (ta + clen - 1 < pos) ? ta + clen - 1 : pos;
                    float qreg[GQ][4], acc[GQ][4], hi[GQ], z[GQ];
                    #pragma unroll
                    for (int g = 0; g < GQ; ++g) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            float qv = ws_q[(size_t)(kvh * GQ + g) * MID_HDIM + lane * 4 + j];
                            if constexpr (!PURE_FP32) qv = tofloat(tobf(qv));
                            qreg[g][j] = qv * MID_ATTN_SCALE;
                            acc[g][j] = 0.0f;
                        }
                        hi[g] = -INFINITY; z[g] = 0.0f;
                    }
                    for (int t = ta; t <= tb; ++t) {
                        const size_t t_slot = is_sliding ? (size_t)(t % MID_WIN) : (size_t)t;
                        const KVT* kt = kc + t_slot * MID_HDIM + lane * 4;
                        const KVT* vt = vc + t_slot * MID_HDIM + lane * 4;
                        float kf[4], vf[4];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) { kf[j] = kv_load<KVT, PURE_FP32>(kt[j]); vf[j] = kv_load<KVT, PURE_FP32>(vt[j]); }
                        float sc[GQ];
                        #pragma unroll
                        for (int g = 0; g < GQ; ++g) {
                            float partial = 0.0f;
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) partial += qreg[g][j] * kf[j];
                            sc[g] = simd_sum(partial);
                        }
                        #pragma unroll
                        for (int g = 0; g < GQ; ++g) {
                            if (sc[g] > hi[g]) {                   // same online update as split-K (no sink)
                                const float f = expf(hi[g] - sc[g]);
                                #pragma unroll
                                for (int j = 0; j < 4; ++j) acc[g][j] = acc[g][j] * f + vf[j];
                                z[g] = z[g] * f + 1.0f; hi[g] = sc[g];
                            } else {
                                const float f = expf(sc[g] - hi[g]);
                                #pragma unroll
                                for (int j = 0; j < 4; ++j) acc[g][j] += f * vf[j];
                                z[g] += f;
                            }
                        }
                    }
                    #pragma unroll
                    for (int g = 0; g < GQ; ++g) {
                        float* pp = part + ((size_t)(kvh * GQ + g) * n_ch + c) * PSTR;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) pp[lane * 4 + j] = acc[g][j];
                        if (lane == 0) { pp[MID_HDIM] = hi[g]; pp[MID_HDIM + 1] = z[g]; }
                    }
                }
                SYNC_ALL();
                for (int h = sg; h < MID_HEADS; h += groups) {  // merge: fold in the sink, normalise
                    const float* ph = part + (size_t)h * n_ch * PSTR;
                    float H = tofloat(sinks[h]);
                    for (int c = 0; c < n_ch; ++c)
                        if (ph[c * PSTR + MID_HDIM + 1] > 0.0f) H = fmaxf(H, ph[c * PSTR + MID_HDIM]);
                    float Z = expf(tofloat(sinks[h]) - H), acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int c = 0; c < n_ch; ++c) {
                        const float zc = ph[c * PSTR + MID_HDIM + 1];
                        if (!(zc > 0.0f)) continue;
                        const float w = expf(ph[c * PSTR + MID_HDIM] - H);
                        Z += zc * w;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) acc[j] += ph[c * PSTR + lane * 4 + j] * w;
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        ws_attn[(size_t)h * MID_HDIM + lane * 4 + j] = acc[j] / Z;
                }
            } else
#elif GEPA_ATTN_SPLIT
            const int n_ctx = pos - t_start + 1;
#if GEPA_ATTN_NSPLIT
            const int n_split = GEPA_ATTN_NSPLIT;
            static_assert(MID_HEADS * GEPA_ATTN_NSPLIT * (MID_HDIM + 2) <= MID_VOCAB, "split partials overflow ws_f_logits");
#else
            const int n_split = groups / MID_HEADS < 16 ? groups / MID_HEADS : 16;
#endif
            if (n_ctx > GEPA_ATTN_SPLIT_MIN && n_split > 1) {   // uniform across workgroups -> the extra barrier is safe
                constexpr int PSTR = MID_HDIM + 2;              // per partial: acc[128], max, denom
                float* const part = ws_f_logits;                // [heads][n_split][PSTR], free until lm_head
                const int clen = (n_ctx + n_split - 1) / n_split;
                auto merge_head = [&](int h) {                  // fold in the sink, normalise (chunk order 0..n_split-1)
                    const float* ph = part + (size_t)h * n_split * PSTR;
                    float H = tofloat(sinks[h]);
                    for (int c = 0; c < n_split; ++c)
                        if (ph[c * PSTR + MID_HDIM + 1] > 0.0f) H = fmaxf(H, ph[c * PSTR + MID_HDIM]);
                    float Z = expf(tofloat(sinks[h]) - H), acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int c = 0; c < n_split; ++c) {
                        const float zc = ph[c * PSTR + MID_HDIM + 1];
                        if (!(zc > 0.0f)) continue;
                        const float w = expf(ph[c * PSTR + MID_HDIM] - H);
                        Z += zc * w;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) acc[j] += ph[c * PSTR + lane * 4 + j] * w;
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        ws_attn[(size_t)h * MID_HDIM + lane * 4 + j] = acc[j] / Z;
                };
#if GEPA_ATTN_LASTMERGE
                static_assert(MID_HEADS * 16 * (MID_HDIM + 2) <= MID_VOCAB - 64, "counters overlap split partials");
                uint32_t* const cnt = reinterpret_cast<uint32_t*>(ws_f_logits + MID_VOCAB - 64);
                auto arrive = [&](int u_) {
                    __threadfence();                                // release this unit's partial (agent scope)
                    uint32_t old = 0u;
                    if (lane == 0)
                        old = __hip_atomic_fetch_add(cnt + u_ / n_split, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
                    old = __shfl_sync(0xffffffffffffffffull, old, 0);
                    if (old == (uint32_t)(n_split - 1)) {           // last arrival for this head
                        __threadfence();                            // acquire: see every chunk's partial
                        merge_head(u_ / n_split);
                    }
                };
#endif
#if GEPA_ATTN_LDS
                if (groups == MID_HEADS * n_split && MID_HEADS / MID_NKV == 8) {   // uniform: every workgroup takes this branch
                    constexpr int LP = 8, LW = 4, LD = MID_HDIM / LW;
                    constexpr int TT = 16384 / (2 * MID_HDIM * (int)sizeof(KVT));     // tile positions: 32 (bf16) / 16 (fp32)
                    constexpr int RW = MID_HDIM * (int)sizeof(KVT) / 16;               // uint4 words per K or V row
                    static_assert(TT % LP == 0, "tile must hold whole LP groups");
                    KVT* const tk = reinterpret_cast<KVT*>(q4_gen_pool(0));            // [TT][HDIM]
                    KVT* const tv = tk + TT * MID_HDIM;                                // [TT][HDIM]
                    const int wq = tid >> 5, pg = lane / LW, sl = lane % LW;
                    const int kvh = group / n_split, c = group % n_split, h = kvh * 8 + wq, u = h * n_split + c;
                    const KVT* kc = l_kv + (0 * MID_NKV + (size_t)kvh) * l_cap * MID_HDIM;
                    const KVT* vc = l_kv + (1 * MID_NKV + (size_t)kvh) * l_cap * MID_HDIM;
                    float q16[LD], a16[LD];
                    #pragma unroll
                    for (int j = 0; j < LD; ++j) {
                        float qv = ws_q[(size_t)h * MID_HDIM + sl * LD + j];
                        if constexpr (!PURE_FP32) qv = tofloat(tobf(qv));
                        q16[j] = qv * MID_ATTN_SCALE;
                        a16[j] = 0.0f;
                    }
                    float hi = -INFINITY, z = 0.0f;
                    const int ta = t_start + c * clen;
                    const int tb = (ta + clen - 1 < pos) ? ta + clen - 1 : pos;
                    __syncthreads();                                   // gemv pool readers of this workgroup are done
                    for (int t0 = ta; t0 <= tb; t0 += TT) {
                        const int nt = (tb - t0 + 1 < TT) ? tb - t0 + 1 : TT;
                        for (int i = tid; i < 2 * TT * RW; i += 256) {  // cooperative copy, one uint4 per thread per step
                            const int sel = i / (TT * RW), r = (i / RW) % TT, wd = i % RW;
                            if (r < nt) {
                                const int t = t0 + r;
                                const size_t slot = is_sliding ? (size_t)(t % MID_WIN) : (size_t)t;
                                const uint4* src = reinterpret_cast<const uint4*>((sel ? vc : kc) + slot * MID_HDIM) + wd;
                                reinterpret_cast<uint4*>((sel ? tv : tk) + (size_t)r * MID_HDIM)[wd] = *src;
                            }
                        }
                        __syncthreads();
                        for (int r0 = 0; r0 < nt; r0 += LP) {           // same position order as the LOP=8 path
                            const int r = r0 + pg;
                            const bool live = r < nt;
                            const int rr = live ? r : nt - 1;
                            float kf[LD], vf[LD];
                            #pragma unroll
                            for (int j = 0; j < LD; ++j) {
                                kf[j] = kv_load<KVT, PURE_FP32>(tk[(size_t)rr * MID_HDIM + sl * LD + j]);
                                vf[j] = kv_load<KVT, PURE_FP32>(tv[(size_t)rr * MID_HDIM + sl * LD + j]);
                            }
                            float partial = 0.0f;
                            #pragma unroll
                            for (int j = 0; j < LD; ++j) partial += q16[j] * kf[j];
                            #pragma unroll
                            for (int off = LW / 2; off >= 1; off >>= 1)
                                partial += __shfl_xor_sync(0xffffffffffffffffull, partial, off);
                            if (live) {
                                if (partial > hi) {
                                    const float f = expf(hi - partial);
                                    #pragma unroll
                                    for (int j = 0; j < LD; ++j) a16[j] = a16[j] * f + vf[j];
                                    z = z * f + 1.0f; hi = partial;
                                } else {
                                    const float f = expf(partial - hi);
                                    #pragma unroll
                                    for (int j = 0; j < LD; ++j) a16[j] += f * vf[j];
                                    z += f;
                                }
                            }
                        }
                        __syncthreads();                               // tile fully consumed before the next copy
                    }
                    {   // merge the LP position groups, same as the LOP path
                        float H = hi;
                        #pragma unroll
                        for (int off = LW; off < 32; off <<= 1) H = fmaxf(H, __shfl_xor_sync(0xffffffffffffffffull, H, off));
                        const float w = (z > 0.0f) ? expf(hi - H) : 0.0f;
                        float zz = z * w;
                        #pragma unroll
                        for (int off = LW; off < 32; off <<= 1) zz += __shfl_xor_sync(0xffffffffffffffffull, zz, off);
                        float* pp = part + (size_t)u * PSTR;
                        #pragma unroll
                        for (int j = 0; j < LD; ++j) {
                            float a_ = a16[j] * w;
                            #pragma unroll
                            for (int off = LW; off < 32; off <<= 1) a_ += __shfl_xor_sync(0xffffffffffffffffull, a_, off);
                            if (pg == 0) pp[sl * LD + j] = a_;
                        }
                        if (lane == 0) { pp[MID_HDIM] = H; pp[MID_HDIM + 1] = zz; }
                    }
#if GEPA_ATTN_LASTMERGE
                    arrive(u);
#endif
                } else
#endif
                for (int u = sg; u < MID_HEADS * n_split; u += groups) {
                    const int h = u / n_split, c = u % n_split;
                    const int kvh = h / (MID_HEADS / MID_NKV);
                    const KVT* kc = l_kv + (0 * MID_NKV + (size_t)kvh) * l_cap * MID_HDIM;
                    const KVT* vc = l_kv + (1 * MID_NKV + (size_t)kvh) * l_cap * MID_HDIM;
                    float qreg[4], acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float qv = ws_q[(size_t)h * MID_HDIM + lane * 4 + j];
                        if constexpr (!PURE_FP32) qv = tofloat(tobf(qv));
                        qreg[j] = qv * MID_ATTN_SCALE;
                    }
#if GEPA_ATTN_LOP
                    // candidate (1) lanes-over-positions: LP positions per iteration; lane group pg = lane / LW takes t + pg,
                    // sub-lane sl = lane % LW holds dims [sl*LD, sl*LD+LD): log2(LW)-step reduction instead of 5; groups merged
                    // at the end into the same per-(head, chunk) partial. iter 3: GEPA_ATTN_LOP_PF prefetches the next
                    // iteration's K/V into registers; GEPA_ATTN_LOP_UNROLL unrolls the position loop.
                    constexpr int LP = GEPA_ATTN_LOP, LW = 32 / LP, LD = MID_HDIM / LW;
                    static_assert(LP == 4 || LP == 8, "GEPA_ATTN_LOP must be 4 or 8");
                    const int pg = lane / LW, sl = lane % LW;
                    float q16[LD], a16[LD];
                    #pragma unroll
                    for (int j = 0; j < LD; ++j) {
                        float qv = ws_q[(size_t)h * MID_HDIM + sl * LD + j];
                        if constexpr (!PURE_FP32) qv = tofloat(tobf(qv));
                        q16[j] = qv * MID_ATTN_SCALE;
                        a16[j] = 0.0f;
                    }
                    float hi = -INFINITY, z = 0.0f;
                    const int ta = t_start + c * clen;
                    const int tb = (ta + clen - 1 < pos) ? ta + clen - 1 : pos;
                    auto slot_of = [&](int t0) -> size_t {
                        const int t = t0 + pg;
                        const int tt = t <= tb ? t : tb;
                        return is_sliding ? (size_t)(tt % MID_WIN) : (size_t)tt;
                    };
#if GEPA_ATTN_LOP_PF
                    float kn[LD], vn[LD];
                    {
                        const size_t s0 = slot_of(ta);
                        #pragma unroll
                        for (int j = 0; j < LD; ++j) {
                            kn[j] = kv_load<KVT, PURE_FP32>(kc[s0 * MID_HDIM + sl * LD + j]);
                            vn[j] = kv_load<KVT, PURE_FP32>(vc[s0 * MID_HDIM + sl * LD + j]);
                        }
                    }
#endif
#if GEPA_ATTN_LOP_UNROLL > 1
                    #pragma unroll GEPA_ATTN_LOP_UNROLL
#endif
                    for (int t0 = ta; t0 <= tb; t0 += LP) {
                        const bool live = t0 + pg <= tb;
                        float kf[LD], vf[LD];
#if GEPA_ATTN_LOP_PF
                        #pragma unroll
                        for (int j = 0; j < LD; ++j) { kf[j] = kn[j]; vf[j] = vn[j]; }
                        if (t0 + LP <= tb) {                       // uniform: next iteration exists
                            const size_t s1 = slot_of(t0 + LP);
                            #pragma unroll
                            for (int j = 0; j < LD; ++j) {
                                kn[j] = kv_load<KVT, PURE_FP32>(kc[s1 * MID_HDIM + sl * LD + j]);
                                vn[j] = kv_load<KVT, PURE_FP32>(vc[s1 * MID_HDIM + sl * LD + j]);
                            }
                        }
#else
                        {
                            const size_t s0 = slot_of(t0);
                            #pragma unroll
                            for (int j = 0; j < LD; ++j) {
                                kf[j] = kv_load<KVT, PURE_FP32>(kc[s0 * MID_HDIM + sl * LD + j]);
                                vf[j] = kv_load<KVT, PURE_FP32>(vc[s0 * MID_HDIM + sl * LD + j]);
                            }
                        }
#endif
                        float partial = 0.0f;
                        #pragma unroll
                        for (int j = 0; j < LD; ++j) partial += q16[j] * kf[j];
                        #pragma unroll
                        for (int off = LW / 2; off >= 1; off >>= 1)
                            partial += __shfl_xor_sync(0xffffffffffffffffull, partial, off);
                        if (live) {
                            if (partial > hi) {
                                const float f = expf(hi - partial);
                                #pragma unroll
                                for (int j = 0; j < LD; ++j) a16[j] = a16[j] * f + vf[j];
                                z = z * f + 1.0f; hi = partial;
                            } else {
                                const float f = expf(partial - hi);
                                #pragma unroll
                                for (int j = 0; j < LD; ++j) a16[j] += f * vf[j];
                                z += f;
                            }
                        }
                    }
                    {   // merge the LP position groups (lanes sl, sl+LW, ... hold the same dims)
                        float H = hi;
                        #pragma unroll
                        for (int off = LW; off < 32; off <<= 1) H = fmaxf(H, __shfl_xor_sync(0xffffffffffffffffull, H, off));
                        const float w = (z > 0.0f) ? expf(hi - H) : 0.0f;
                        float zz = z * w;
                        #pragma unroll
                        for (int off = LW; off < 32; off <<= 1) zz += __shfl_xor_sync(0xffffffffffffffffull, zz, off);
                        float* pp = part + (size_t)u * PSTR;
                        #pragma unroll
                        for (int j = 0; j < LD; ++j) {
                            float a_ = a16[j] * w;
                            #pragma unroll
                            for (int off = LW; off < 32; off <<= 1) a_ += __shfl_xor_sync(0xffffffffffffffffull, a_, off);
                            if (pg == 0) pp[sl * LD + j] = a_;
                        }
                        if (lane == 0) { pp[MID_HDIM] = H; pp[MID_HDIM + 1] = zz; }
                    }
                    (void)qreg; (void)acc;
#else
                    float hi = -INFINITY, z = 0.0f;
                    const int ta = t_start + c * clen;
                    const int tb = (ta + clen - 1 < pos) ? ta + clen - 1 : pos;
                    for (int t = ta; t <= tb; ++t) {
                        size_t t_slot = is_sliding ? (size_t)(t % MID_WIN) : (size_t)t;
                        const KVT* kt = kc + t_slot * MID_HDIM + lane * 4;
                        const KVT* vt = vc + t_slot * MID_HDIM + lane * 4;
                        float partial = 0.0f;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) partial += qreg[j] * kv_load<KVT, PURE_FP32>(kt[j]);
                        float score = simd_sum(partial);
                        if (score > hi) {                       // same online update as the 1-warp path, no sink
                            float f = expf(hi - score);         // hi = -inf -> 0
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) acc[j] = acc[j] * f + kv_load<KVT, PURE_FP32>(vt[j]);
                            z = z * f + 1.0f;
                            hi = score;
                        } else {
                            float f = expf(score - hi);
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) acc[j] = acc[j] + f * kv_load<KVT, PURE_FP32>(vt[j]);
                            z = z + f;
                        }
                    }
                    float* pp = part + (size_t)u * PSTR;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) pp[lane * 4 + j] = acc[j];
                    if (lane == 0) { pp[MID_HDIM] = hi; pp[MID_HDIM + 1] = z; }
#endif
#if GEPA_ATTN_LASTMERGE
                    arrive(u);
#endif
                }
#if !GEPA_ATTN_LASTMERGE
                SYNC_ALL();
#if GEPA_ATTN_NSPLIT || GEPA_ATTN_LDS
                // lever paths: lambda merge (capture lifetimes differ; hot-loop RA handled separately)
                for (int h = sg; h < MID_HEADS; h += groups) merge_head(h);
#else
                // rec-bhz: ORIGINAL iter-4 inline merge text — merge_head's captures extend value lifetimes across
                // the u-loop and hipcc RA turns that into 24 extra scratch ops in the hot path (-12% ctx8192).
                // Keep the lever-off SPLIT path textually identical to the measured 167.26 build.
                for (int h = sg; h < MID_HEADS; h += groups) {  // merge: fold in the sink, normalise
                    const float* ph = part + (size_t)h * n_split * PSTR;
                    float H = tofloat(sinks[h]);
                    for (int c = 0; c < n_split; ++c)
                        if (ph[c * PSTR + MID_HDIM + 1] > 0.0f) H = fmaxf(H, ph[c * PSTR + MID_HDIM]);
                    float Z = expf(tofloat(sinks[h]) - H), acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int c = 0; c < n_split; ++c) {
                        const float zc = ph[c * PSTR + MID_HDIM + 1];
                        if (!(zc > 0.0f)) continue;
                        const float w = expf(ph[c * PSTR + MID_HDIM] - H);
                        Z += zc * w;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) acc[j] += ph[c * PSTR + lane * 4 + j] * w;
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        ws_attn[(size_t)h * MID_HDIM + lane * 4 + j] = acc[j] / Z;
                }
#endif
#endif
            } else
#endif
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
            if (GEPA_PF > 0 && sg >= MID_HEADS)   // idle in attention: warm the O-projection prefix
                pf_warm(Q_W(qbase + 3), (size_t)MID_HID * (MID_QDIM / 2), nullptr, 0, nullptr, 0, 0, sg - MID_HEADS, lane, pf_sink);
            SYNC_ALL();
            MID_DUMP(MID_DUMP_ATTN, ws_attn, MID_QDIM);

            // ---- 7. O projection (K=4096 -> N=2048) + residual ----
#if GEPA_RESID_FUSE
            q4_block512<MID_QDIM>(Q_W(qbase + 3), Q_S(qbase + 3, MID_HID, MID_QDIM), Q_B(qbase + 3, MID_HID, MID_QDIM),
                                  ws_attn, nullptr, ws_x, MID_HID, lane, sg, groups, true);
            SYNC_ALL();
#elif GEPA_OSPLIT
            static_assert(MID_QDIM == 2 * MID_HID, "O-proj split-K assumes K = 2 x 2048");
            if (tid == 0) {
                const uint8_t* ow = Q_W(qbase + 3);
                const bf16* os = Q_S(qbase + 3, MID_HID, MID_QDIM);
                const bf16* ob = Q_B(qbase + 3, MID_HID, MID_QDIM);
                sh_seg[0] = {ow, os, ob, ws_o, MID_HID, 2};
                sh_seg[1] = {ow + 1024, os + 32, ob + 32, ws_down, MID_HID, 2};
            }
            __syncthreads();
            flint_q4_gemv_op<MID_HID, false, 0, 2>(nullptr, nullptr, nullptr, ws_attn, nullptr, nullptr, 0, lane, sg, groups,
                                                   sh_seg, 1);
            flint_q4_gemv_op<MID_HID, false, 0, 2>(nullptr, nullptr, nullptr, ws_attn + MID_HID, nullptr, nullptr, 0, lane, sg,
                                                   groups, sh_seg + 1, 1);
            SYNC_ALL();
            for (int i = gtid; i < MID_HID; i += gthreads)
                ws_x[i] += ws_o[i] + ws_down[i];
            SYNC_ALL();
#else
            q4_block512<MID_QDIM>(Q_W(qbase + 3), Q_S(qbase + 3, MID_HID, MID_QDIM), Q_B(qbase + 3, MID_HID, MID_QDIM),
                                  ws_attn, nullptr, ws_o, MID_HID, lane, sg, groups);
            SYNC_ALL();
#if GEPA_DUMP >= 2
            MID_DUMP(MID_DUMP_O, ws_o, MID_HID);
#endif
#if GEPA_RACE_EXP == 3
            SYNC_ALL();                                   // rec-3cr E3: double barrier before the residual
#endif
            for (int i = gtid; i < MID_HID; i += gthreads)
#if GEPA_RACE_EXP == 4
                ws_x[i] += __hip_atomic_load(ws_o + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);   // E4: bypass L0/GL1
#else
                ws_x[i] += ws_o[i];
#endif
            SYNC_ALL();
#endif

            MID_DUMP(MID_DUMP_XA, ws_x, MID_HID);
            // ---- 8. Post-attention RMSNorm -> sh_xn ----
            mid_rmsnorm_2048(ws_x, post_w, sh_xn, sh_red);

            // ---- 9. MLP (Layer 0 Dense vs Layers 1..7 MoE) ----
            if (layer < MID_N_DENSE) {
                if (group == 0 && tid == 0 && out_chosen_experts != nullptr) {
                    int* exp_dst = out_chosen_experts + ((size_t)lp * MID_NL + layer) * MID_TOPK;
                    for (int t = 0; t < MID_TOPK; ++t) exp_dst[t] = -1;
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
#if GEPA_RESID_FUSE
                q4_block512<MID_DENSE_I>(Q_W(qbase + 6), Q_S(qbase + 6, MID_HID, MID_DENSE_I), Q_B(qbase + 6, MID_HID, MID_DENSE_I),
                                         ws_f_act, nullptr, ws_x, MID_HID, lane, sg, groups, true);
                SYNC_ALL();
#else
                q4_block512<MID_DENSE_I>(Q_W(qbase + 6), Q_S(qbase + 6, MID_HID, MID_DENSE_I), Q_B(qbase + 6, MID_HID, MID_DENSE_I),
                                         ws_f_act, nullptr, ws_down, MID_HID, lane, sg, groups);
                SYNC_ALL();
                for (int i = gtid; i < MID_HID; i += gthreads)
                    ws_x[i] += ws_down[i];
                SYNC_ALL();
#endif
                MID_DUMP(MID_DUMP_XM, ws_x, MID_HID);
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
                if (GEPA_PF > 0 && sg >= MID_NEXP)    // idle in the router: warm the shared gate/up prefix
                    pf_warm(Q_W(qbase + 7), (size_t)MID_SH_I * (MID_HID / 2), Q_W(qbase + 8), (size_t)MID_SH_I * (MID_HID / 2),
                            nullptr, 0, 0, sg - MID_NEXP, lane, pf_sink);
                SYNC_ALL();
                MID_DUMP(MID_DUMP_SC, ws_f_scores, MID_NEXP);
#if GEPA_TOPK_PER_WG
                __shared__ int   sh_sel[MID_TOPK];
                __shared__ float sh_wsel[MID_TOPK];
#define GEPA_SEL(t) sh_sel[t]
#define GEPA_WSEL(t) sh_wsel[t]
#else
#define GEPA_SEL(t) ws_i_sel[t]
#define GEPA_WSEL(t) ws_f_wsel[t]
#endif
                if (GEPA_TOPK_PER_WG ? tid == 0 : (group == 0 && tid == 0)) {
                    // top-K by insertion, strict '>' (an equal later expert never displaces an
                    // earlier one): for TOPK == 2 this is exactly the original two-slot update.
                    int   ti[MID_TOPK];
                    float tv[MID_TOPK];
                    for (int t = 0; t < MID_TOPK; ++t) { ti[t] = 0; tv[t] = -1e30f; }
                    for (int e = 0; e < MID_NEXP; ++e) {
                        float choice = ws_f_scores[e] + tofloat(rb_bf[e]);
                        if (!(choice > tv[MID_TOPK - 1])) continue;
                        int p = MID_TOPK - 1;
                        while (p > 0 && choice > tv[p - 1]) { tv[p] = tv[p - 1]; ti[p] = ti[p - 1]; --p; }
                        tv[p] = choice; ti[p] = e;
                    }
                    for (int t = 0; t < MID_TOPK; ++t) GEPA_SEL(t) = ti[t];
                    if (GEPA_TOPK_PER_WG && group == 0) for (int t = 0; t < MID_TOPK; ++t) ws_i_sel[t] = ti[t];
                    if (group == 0 && out_chosen_experts != nullptr) {
                        int* exp_dst = out_chosen_experts + ((size_t)lp * MID_NL + layer) * MID_TOPK;
                        for (int t = 0; t < MID_TOPK; ++t) exp_dst[t] = ti[t];
                    }
                    if constexpr (MID_TOPK == 2) {
                        float s0 = ws_f_scores[ti[0]], s1 = ws_f_scores[ti[1]];
                        float wsum = s0 + s1 + 1e-20f;
                        GEPA_WSEL(0) = s0 / wsum;
                        GEPA_WSEL(1) = s1 / wsum;
                    } else {
                        float wsum = 0.0f;
                        for (int t = 0; t < MID_TOPK; ++t) wsum += ws_f_scores[ti[t]];
                        wsum += 1e-20f;
                        for (int t = 0; t < MID_TOPK; ++t) GEPA_WSEL(t) = ws_f_scores[ti[t]] / wsum;
                    }
                    if (GEPA_TOPK_PER_WG && group == 0) for (int t = 0; t < MID_TOPK; ++t) ws_f_wsel[t] = GEPA_WSEL(t);
                }
                if (GEPA_PF > 0 && !GEPA_TOPK_PER_WG && sg >= 8)   // idle during top-K: continue the shared window
                    pf_warm(Q_W(qbase + 7), (size_t)MID_SH_I * (MID_HID / 2), Q_W(qbase + 8), (size_t)MID_SH_I * (MID_HID / 2),
                            nullptr, 0, (size_t)(groups > MID_NEXP ? groups - MID_NEXP : 0) * GEPA_PF, sg - 8, lane, pf_sink);
#if GEPA_TOPK_PER_WG
                __syncthreads();
#else
                SYNC_ALL();
#endif

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

                // rec-9t8 lever 1: batched MoE phases. All TOPK routed experts + the shared expert run
                // gate/up in ONE phase, act in one, down in one (was 3 barriers per expert + 3 shared).
                // Per-row gemvs and the act formula are unchanged; only the barrier schedule differs.
                // ws_gate/ws_up/ws_f_act are laid out [t * MOE_I] for routed t, [TOPK * MOE_I] shared.
                static_assert(MID_TOPK * MID_MOE_I + MID_SH_I <= MID_FFN_MAX, "batched MoE ws layout");
                constexpr int MOE_SH0 = MID_TOPK * MID_MOE_I;      // shared-expert slot offset
                constexpr int MOE_ACT_N = MOE_SH0 + MID_SH_I;
                // rec-9t8 3b: all routed gate/up + shared gate/up as one segmented K=2048 gemv
                // GEPA_PF: shared expert first in the quad stream (it is the prefetched window); rows/arithmetic unchanged
                constexpr int SEG_R0 = GEPA_PF > 0 ? 2 : 0, SEG_SH = GEPA_PF > 0 ? 0 : 2 * MID_TOPK;
                if (tid < MID_TOPK) {
                    int t = tid, e = GEPA_SEL(t);
                    // Expert strides: bytes for packed uint8*, ELEMENTS for bf16* scales/biases
                    size_t gw_off = (size_t)e * MID_MOE_I * (MID_HID / 2);
                    size_t gs_off = (size_t)e * MID_MOE_I * (MID_HID / 64);
                    sh_seg[SEG_R0 + 2 * t]     = {sw_gw + gw_off, sw_gs + gs_off, sw_gb + gs_off, ws_gate + t * MID_MOE_I, MID_MOE_I};
                    sh_seg[SEG_R0 + 2 * t + 1] = {sw_uw + gw_off, sw_us + gs_off, sw_ub + gs_off, ws_up + t * MID_MOE_I, MID_MOE_I};
                } else if (tid == MID_TOPK) {
                    // Shared expert (SH_I = 1024, HID = 2048)
                    sh_seg[SEG_SH]     = {Q_W(qbase + 7), Q_S(qbase + 7, MID_SH_I, MID_HID), Q_B(qbase + 7, MID_SH_I, MID_HID),
                                                ws_gate + MOE_SH0, MID_SH_I};
                    sh_seg[SEG_SH + 1] = {Q_W(qbase + 8), Q_S(qbase + 8, MID_SH_I, MID_HID), Q_B(qbase + 8, MID_SH_I, MID_HID),
                                                ws_up + MOE_SH0, MID_SH_I};
                }
                __syncthreads();
                flint_q4_gemv_op<MID_HID, false, GEPA_S2_MOE>(nullptr, nullptr, nullptr, sh_xn, nullptr, nullptr, 0, lane, sg,
                                          groups, sh_seg, 2 * MID_TOPK + 2);
                SYNC_ALL();
#if GEPA_ACT_FOLD
                static_assert(MOE_ACT_N <= MID_HID, "act fold reuses sh_xn[MID_HID]");
                for (int i = tid; i < MOE_ACT_N; i += 256) {
                    if constexpr (PURE_FP32) {
                        sh_xn[i] = silu_f(ws_gate[i]) * ws_up[i];
                    } else {
                        bf16 act = tobf(silu_f(tofloat(tobf(ws_gate[i]))) * tofloat(tobf(ws_up[i])));
                        sh_xn[i] = tofloat(act);
                    }
                }
                __syncthreads();
                const float* act_src = sh_xn;
#elif GEPA_ACTX
                static_assert(!GEPA_DOWNCAT, "ACTX and DOWNCAT are exclusive");
                const float* act_src = nullptr;   // down gemvs compute act(gate, up) in their x-load
#else
                for (int i = gtid; i < MOE_ACT_N; i += gthreads) {
                    if constexpr (PURE_FP32) {
                        ws_f_act[i] = silu_f(ws_gate[i]) * ws_up[i];
                    } else {
                        bf16 act = tobf(silu_f(tofloat(tobf(ws_gate[i]))) * tofloat(tobf(ws_up[i])));
                        ws_f_act[i] = tofloat(act);
                    }
                    if (GEPA_DOWNCAT && i < MOE_SH0) ws_f_act[i] *= GEPA_WSEL(i / MID_MOE_I);   // fold routing weight
                }
                if (GEPA_PF > 0 && sg * 32 >= MOE_ACT_N) {   // idle in act: warm down weights in consumption order
                    constexpr size_t DB = (size_t)MID_HID * (MID_MOE_I / 2);
                    const uint8_t* d0 = sw_dw + (size_t)GEPA_SEL(0) * DB;
                    const uint8_t* d1 = MID_TOPK > 1 ? sw_dw + (size_t)GEPA_SEL(1) * DB : nullptr;
                    pf_warm(d0, DB, d1, MID_TOPK > 1 ? DB : 0, Q_W(qbase + 9), (size_t)MID_HID * (MID_SH_I / 2), 0,
                            sg - MOE_ACT_N / 32, lane, pf_sink);
                }
                SYNC_ALL();
                const float* act_src = ws_f_act;
#endif
#if GEPA_DOWNCAT
                static_assert(!GEPA_ACT_FOLD, "DOWNCAT uses the ws_f_act path");
                static_assert(MID_TOPK == 2 && MID_MOE_I == 512 && MID_SH_I == 1024, "DOWNCAT: virtual K = 2x512 + 1024");
                {
                    Q4Cat c;
                    #pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        const int e = GEPA_SEL(t);
                        c.w[t] = sw_dw + (size_t)e * MID_HID * (MID_MOE_I / 2);
                        c.s[t] = sw_ds + (size_t)e * MID_HID * (MID_MOE_I / 64);
                        c.b[t] = sw_db + (size_t)e * MID_HID * (MID_MOE_I / 64);
                        c.rb[t] = MID_MOE_I / 2; c.gpr[t] = MID_MOE_I / 64; c.k0[t] = 0;
                    }
                    #pragma unroll
                    for (int j = 2; j < 4; ++j) {
                        c.w[j] = Q_W(qbase + 9);
                        c.s[j] = Q_S(qbase + 9, MID_HID, MID_SH_I);
                        c.b[j] = Q_B(qbase + 9, MID_HID, MID_SH_I);
                        c.rb[j] = MID_SH_I / 2; c.gpr[j] = MID_SH_I / 64; c.k0[j] = (j - 2) * 512;
                    }
                    q4_gemv_cat2048(c, act_src, ws_down, MID_HID, lane, sg, groups);
                }
                SYNC_ALL();
                for (int i = gtid; i < MID_HID; i += gthreads) ws_x[i] += ws_down[i];   // weights already folded
                if (false) {   // original down + weighted combine (dead under DOWNCAT)
#endif
                for (int t = 0; t < MID_TOPK; ++t) {
                    int e = GEPA_SEL(t);
                    size_t dw_off = (size_t)e * MID_HID * (MID_MOE_I / 2);
                    size_t ds_off = (size_t)e * MID_HID * (MID_MOE_I / 64);
#if GEPA_ACTX
                    flint_q4_gemv_op<MID_MOE_I, false, 0, 1, MID_FFN_MAX, PURE_FP32>(
                        sw_dw + dw_off, sw_ds + ds_off, sw_db + ds_off, ws_gate + t * MID_MOE_I, nullptr, ws_down + t * MID_HID,
                        MID_HID, lane, sg, groups);
#else
                    q4_block512<MID_MOE_I>(sw_dw + dw_off, sw_ds + ds_off, sw_db + ds_off,
                                           act_src + t * MID_MOE_I, nullptr, ws_down + t * MID_HID, MID_HID, lane, sg, groups);
#endif
                }
#if GEPA_ACTX
                flint_q4_gemv_op<MID_SH_I, false, 0, 1, MID_FFN_MAX, PURE_FP32>(
                    Q_W(qbase + 9), Q_S(qbase + 9, MID_HID, MID_SH_I), Q_B(qbase + 9, MID_HID, MID_SH_I), ws_gate + MOE_SH0,
                    nullptr, ws_down + MID_TOPK * MID_HID, MID_HID, lane, sg, groups);
#else
                q4_block512<MID_SH_I>(Q_W(qbase + 9), Q_S(qbase + 9, MID_HID, MID_SH_I), Q_B(qbase + 9, MID_HID, MID_SH_I),
                                      act_src + MOE_SH0, nullptr, ws_down + MID_TOPK * MID_HID, MID_HID, lane, sg, groups);
#endif
                SYNC_ALL();

                if constexpr (MID_TOPK == 2) {
                    float w0 = GEPA_WSEL(0), w1 = GEPA_WSEL(1);
                    for (int i = gtid; i < MID_HID; i += gthreads) {
                        ws_x[i] += w0 * ws_down[i] + w1 * ws_down[MID_HID + i] + ws_down[2 * MID_HID + i];
                    }
                } else {
                    float wt[MID_TOPK];
                    for (int t = 0; t < MID_TOPK; ++t) wt[t] = GEPA_WSEL(t);
                    for (int i = gtid; i < MID_HID; i += gthreads) {
                        float acc = 0.0f;
                        #pragma unroll
                        for (int t = 0; t < MID_TOPK; ++t) acc += wt[t] * ws_down[(size_t)t * MID_HID + i];
                        ws_x[i] += acc + ws_down[(size_t)MID_TOPK * MID_HID + i];
                    }
                }
#if GEPA_DOWNCAT
                }
#endif
                if (GEPA_PF > 0 && sg * 32 >= MID_HID) {     // idle in combine: warm the next weight phase
                    if (layer + 1 < MID_NL) {
                        int nb = MID_OFF_LAYER_Q4_BASE + (layer + 1) * 10;
                        pf_warm(Q_W(nb + 0), (size_t)MID_QDIM * (MID_HID / 2), nullptr, 0, nullptr, 0, 0, sg - MID_HID / 32,
                                lane, pf_sink);
                    } else {
                        pf_warm(Q_W(MID_OFF_LM_HEAD), (size_t)MID_VOCAB * (MID_HID / 2), nullptr, 0, nullptr, 0, 0,
                                sg - MID_HID / 32, lane, pf_sink);
                    }
                }
                SYNC_ALL();
                MID_DUMP(MID_DUMP_XM, ws_x, MID_HID);
            }
        }

        // ---- 10. Final RMSNorm -> sh_xn ----
        mid_rmsnorm_2048(ws_x, B(MID_OFF_FINAL_NORM), sh_xn, sh_red);

        // ---- 11. LM Head Fused INT4 Dequant-Matmul (150144 x 2048, rec-aip) -> ws_f_logits ----
        // Always execute lm_head + argmax at every position so persistent decode32
        // differential cancellation holds identically across prefill and decode steps.
        // Uses the integrated `flint_q4_gemv_op<MID_HID>` fast path (preloaded register
        // input slices + 32-bit packed weight loads over zero-copy layout s2).
        flint_q4_gemv_op<MID_HID, false, GEPA_S2_LMH>(Q_W(MID_OFF_LM_HEAD),
                                  Q_S(MID_OFF_LM_HEAD, MID_VOCAB, MID_HID),
                                  Q_B(MID_OFF_LM_HEAD, MID_VOCAB, MID_HID),
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
        int step_slot = lp - prompt_len + 1;
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
        if (lp == total - 1 && out_last_logits) {
            for (int i = gtid; i < MID_VOCAB; i += gthreads)
                out_last_logits[i] = ws_f_logits[i];
        }
        SYNC_ALL();
    }
    if (GEPA_PF > 0 && n_out < 0 && pf_sink == 0x9e3779b9u) control[3] = pf_sink;   // never true; keeps the loads
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
    int* out_chosen_experts, int start_pos) {
    mid_generate_impl<bf16, false>(
        bank, off, kv_sliding, kv_full, full_cap,
        prompt, prompt_len, out_tokens, n_out,
        out_step_logits, out_last_logits, ws_bf, ws_f, ws_i, control, workers,
        out_chosen_experts, start_pos);
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
    int* out_chosen_experts, int start_pos) {
    mid_generate_impl<bf16, true>(
        bank, off, kv_sliding, kv_full, full_cap,
        prompt, prompt_len, out_tokens, n_out,
        out_step_logits, out_last_logits, ws_bf, ws_f, ws_i, control, workers,
        out_chosen_experts, start_pos);
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
    int* out_chosen_experts, int start_pos) {
    mid_generate_impl<float, true>(
        bank, off, kv_sliding, kv_full, full_cap,
        prompt, prompt_len, out_tokens, n_out,
        out_step_logits, out_last_logits, ws_bf, ws_f, ws_i, control, workers,
        out_chosen_experts, start_pos);
}

// ---- Standalone Fused INT4 Dequant-Matmul Kernel (rec-aip) ----
// Consumes the exact zero-copy checkpoint layout from docs/rocm-model-layer.md s2:
//   w: uint8_t [N, K/2], s: bf16 [N, K/64], b: bf16 [N, K/64], x: bf16 [K] -> out: bf16 or fp32 [N]
// Shares `flint_q4_gemv_op` with `mid_generate_impl` (after staging `x` into shared float[K] for K <= 4096).
extern "C" __global__ __launch_bounds__(256) void flint_q4_gemv_kernel(
    const uint8_t* w, const bf16* s, const bf16* b,
    const bf16* x, bf16* out_bf, float* out_f, int n, int k) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int sg = blockIdx.x * (blockDim.x >> 5) + (tid >> 5);
    int groups = gridDim.x * (blockDim.x >> 5);

    __shared__ __attribute__((aligned(16))) float sh_x[4096];   // 16 B: generic path reads x as float4
    if (k <= 4096) {
        for (int i = tid; i < k; i += 256) sh_x[i] = tofloat(x[i]);
        __syncthreads();
        if (k == 2048) {
            flint_q4_gemv_op<2048>(w, s, b, sh_x, out_bf, out_f, n, lane, sg, groups);
            return;
        } else if (k == 4096) {
            flint_q4_gemv_op<4096>(w, s, b, sh_x, out_bf, out_f, n, lane, sg, groups);
            return;
        } else if (k == 1024) {
            flint_q4_gemv_op<1024>(w, s, b, sh_x, out_bf, out_f, n, lane, sg, groups);
            return;
        } else if (k == 512) {
            flint_q4_gemv_op<512>(w, s, b, sh_x, out_bf, out_f, n, lane, sg, groups);
            return;
        }
    }

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
    (void)hipGetLastError(); // clear any stale sticky error before launch
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
        int lrc = flint_q4_gemv_device(d_w, d_s, d_b, d_x, d_out_bf, d_out_f, n, k, grid_blocks);
        if (lrc != 0) {
            fprintf(stderr, "flint_mid q4 warmup launch failed: %s (%d)\n", hipGetErrorString((hipError_t)lrc), lrc);
            return -2;
        }
    }
    chk_hip(hipDeviceSynchronize(), "q4 warmup sync");

    hipEvent_t ev0, ev1;
    chk_hip(hipEventCreate(&ev0), "ev0");
    chk_hip(hipEventCreate(&ev1), "ev1");
    chk_hip(hipEventRecord(ev0, 0), "ev0 rec");
    for (int i = 0; i < iters; ++i) {
        int lrc = flint_q4_gemv_device(d_w, d_s, d_b, d_x, d_out_bf, d_out_f, n, k, grid_blocks);
        if (lrc != 0) {
            fprintf(stderr, "flint_mid q4 timed launch failed: %s (%d)\n", hipGetErrorString((hipError_t)lrc), lrc);
            return -2;
        }
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
    size_t   bank_bytes;   // no default member initializer: `new MidSession()` must stay value-initialized (all null)
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

extern "C" void mid_session_free(void* handle);

// Compile-time config, so a driver can refuse a .so built for the other model.
extern "C" int mid_config(int* out) {
    out[0] = MID_NL; out[1] = MID_NEXP; out[2] = MID_TOPK; out[3] = MID_N_DENSE;
    out[4] = MID_DENSE_I; out[5] = MID_NUM_OFFSETS; out[6] = MID_N_FULL; out[7] = FLINT_CFG_FULL;
    return 8;
}

extern "C" void mid_session_free(void* handle);

// Bedivere M3 window prep: streamed bank load. create2 allocates the session and copies the offset table (bank
// size = sentinel off[N-1]); the caller then streams the bank file with load_bank in chunks from a small reusable
// host buffer (the 21 GiB full bank is never resident in host RAM). mid_session_create == create2 + one load_bank.
extern "C" void* mid_session_create2(const int64_t* h_off, int max_total);
extern "C" int mid_session_load_bank(void* handle, size_t offset, const void* h_src, size_t nbytes);

extern "C" void* mid_session_create(const uint8_t* h_bank, const int64_t* h_off, int max_total) {
    void* s = mid_session_create2(h_off, max_total);
    if (!s) return nullptr;
    if (mid_session_load_bank(s, 0, h_bank, (size_t)h_off[MID_NUM_OFFSETS - 1]) != 0) {
        mid_session_free(s);
        return nullptr;
    }
    return s;
}

extern "C" int mid_session_load_bank(void* handle, size_t offset, const void* h_src, size_t nbytes) {
    MidSession* s = (MidSession*)handle;
    if (!s || !s->d_bank || offset + nbytes > s->bank_bytes) {
        fprintf(stderr, "flint_mid load_bank: bad range offset=%zu n=%zu bank=%zu\n", offset, nbytes,
                s ? s->bank_bytes : (size_t)0);
        return -1;
    }
    hipError_t e = hipMemcpy(s->d_bank + offset, h_src, nbytes, hipMemcpyHostToDevice);
    if (e != hipSuccess) {
        fprintf(stderr, "flint_mid load_bank: hipMemcpy FAILED at %zu (%s)\n", offset, hipGetErrorString(e));
        return -2;
    }
    return 0;
}

extern "C" void* mid_session_create2(const int64_t* h_off, int max_total) {
    MidSession* s = new MidSession();
    s->max_total = max_total;
    size_t bank_sz = (size_t)h_off[MID_NUM_OFFSETS - 1];
    size_t off_sz  = MID_NUM_OFFSETS * sizeof(int64_t);
    size_t kvs_elems = (size_t)MID_N_SLIDING * 2 * MID_NKV * MID_WIN * MID_HDIM;
    size_t kvf_elems = (size_t)MID_N_FULL * 2 * MID_NKV * (size_t)max_total * MID_HDIM;
    size_t ws_bf_sz = 1024 * sizeof(bf16);
    size_t ws_f_sz  = ((MID_WS_LOGITS_OFF > 32896 ? (size_t)MID_WS_LOGITS_OFF : (size_t)32896) + MID_VOCAB) * sizeof(float);
    size_t ws_i_sz  = 128 * sizeof(int);

    // rec-u9q: a failed allocation or bank copy must not leave a session that launches on garbage
    // (chk_hip only logs). The 21 GiB full-Flint bank is the first allocation that can fail.
    if (hipMalloc(&s->d_bank, bank_sz) != hipSuccess) {
        fprintf(stderr, "flint_mid session_create: hipMalloc bank (%zu bytes) FAILED\n", bank_sz);
        delete s;
        return nullptr;
    }
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

    s->bank_bytes = bank_sz;
    hipError_t co = hipMemcpy(s->d_off, h_off, off_sz, hipMemcpyHostToDevice);
    hipError_t ce = hipGetLastError();
    if (co != hipSuccess || ce != hipSuccess || !s->d_chosen_experts || !s->d_logits) {
        fprintf(stderr, "flint_mid session_create: offset copy or allocation FAILED (%s)\n",
                hipGetErrorString(co != hipSuccess ? co : ce));
        mid_session_free(s);
        return nullptr;
    }
    return s;
}

// mode:
//   0 = production bf16-KV build (`mid_generate`, KVT=bf16, PURE_FP32=false)
//   1 = pure-fp32 compute + bf16-KV build (`mid_generate_fp32_bf16kv`, KVT=bf16, PURE_FP32=true)
//   2 = pure-fp32 compute + fp32-KV validation build (`mid_generate_fp32kv`, KVT=float, PURE_FP32=true)
// h_out_chosen_experts (optional, default nullptr):
//   int32 buffer of shape [prompt_len + n_out, MID_NL, MID_TOPK] (mid-8L: 8 x 2; full Flint: 49 x 6).
//   When nullptr, costs zero device work (null pointer passed to kernel).
// Returns 0 on success (with control[2] == 0), nonzero on invalid arguments, HIP error, or barrier abort.
// Bedivere M3.1: KV continuation. start_pos = absolute position of prompt[0]; the session's KV for positions
// [0, start_pos) must have been written by earlier launches of the SAME session and mode. Continuation protocol:
// next launch = prompt {last token of the previous launch}, start_pos = prev_start + prev_prompt_len + prev_n_out - 1
// (that position is re-run with identical inputs, rewriting identical KV).
extern "C" int mid_session_run_ex3(void* handle, const int* h_prompt, int prompt_len,
                                   int* h_out_tokens, int n_out,
                                   float* h_out_step_logits, float* h_out_last_logits,
                                   int workers, int mode, uint32_t* out_control,
                                   int* h_out_chosen_experts, int start_pos) {
    MidSession* s = (MidSession*)handle;
    if (!s) return -1;
    // R3: reject invalid workers loudly (no silent default to 16; enforce compiled cap <= 64)
    if (workers < 1 || workers > MID_MAX_WORKERS) {
        fprintf(stderr, "flint_mid error: workers=%d outside valid range [1, %d]\n", workers, MID_MAX_WORKERS);
        return -2;
    }
    // A5(a): enforce full-layer KV capacity >= start + P + N
    if (start_pos < 0 || start_pos + prompt_len + n_out > s->max_total) {
        fprintf(stderr, "flint_mid error: start_pos(%d)+prompt_len(%d)+n_out(%d) > max_total(%d)\n",
                start_pos, prompt_len, n_out, s->max_total);
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

    // Clear any prior sticky HIP error so hipGetLastError() reflects this launch only
    (void)hipGetLastError();

    const void* fn = nullptr;
    void* k_kv_s = nullptr;
    void* k_kv_f = nullptr;
    if (mode == 2) {
        fn = (const void*)mid_generate_fp32kv;
        k_kv_s = (void*)s->d_kv_sliding_f32;
        k_kv_f = (void*)s->d_kv_full_f32;
    } else if (mode == 1) {
        fn = (const void*)mid_generate_fp32_bf16kv;
        k_kv_s = (void*)s->d_kv_sliding;
        k_kv_f = (void*)s->d_kv_full;
    } else {
        fn = (const void*)mid_generate;
        k_kv_s = (void*)s->d_kv_sliding;
        k_kv_f = (void*)s->d_kv_full;
    }

    void* kargs[] = {
        (void*)&s->d_bank,
        (void*)&s->d_off,
        (void*)&k_kv_s,
        (void*)&k_kv_f,
        (void*)&s->max_total,
        (void*)&s->d_prompt,
        (void*)&prompt_len,
        (void*)&s->d_out,
        (void*)&n_out,
        (void*)&d_step,
        (void*)&d_last,
        (void*)&s->d_ws_bf,
        (void*)&s->d_ws_f,
        (void*)&s->d_ws_i,
        (void*)&s->d_control,
        (void*)&workers,
        (void*)&d_exp,
        (void*)&start_pos,
    };
    hipError_t launch_err = hipLaunchKernel(fn, dim3(workers), dim3(256), kargs, 0, 0);
    hipError_t last_err   = hipGetLastError();
    if (launch_err != hipSuccess || last_err != hipSuccess) {
        hipError_t err = (launch_err != hipSuccess) ? launch_err : last_err;
        fprintf(stderr, "flint_mid mid_generate launch error (mode=%d, workers=%d): %s (%d)\n",
                mode, workers, hipGetErrorString(err), (int)err);
        return -6;
    }

    hipError_t sync_err = hipDeviceSynchronize();
    if (sync_err != hipSuccess) {
        fprintf(stderr, "flint_mid mid_generate sync error: %s\n", hipGetErrorString(sync_err));
        return -4;
    }
    // R2 & R5 & rec-tan.5: always read back control[0..2] and check:
    //   1) control[2] == 0 (no barrier spin-timeout abort)
    //   2) control[1] > 0 when prompt_len + n_out > 0 (kernel actually executed barriers; guards against no-op launch)
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
    if ((prompt_len + n_out) > 0 && h_ctrl[1] == 0u) {
        fprintf(stderr, "flint_mid LAUNCH DID NOT EXECUTE: control[1]=0 (phase=0) after synchronize (workers=%d, mode=%d)\n",
                workers, mode);
        return -7;
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

extern "C" int mid_session_run_ex2(void* handle, const int* h_prompt, int prompt_len,
                                   int* h_out_tokens, int n_out,
                                   float* h_out_step_logits, float* h_out_last_logits,
                                   int workers, int mode, uint32_t* out_control,
                                   int* h_out_chosen_experts) {
    return mid_session_run_ex3(handle, h_prompt, prompt_len, h_out_tokens, n_out, h_out_step_logits,
                               h_out_last_logits, workers, mode, out_control, h_out_chosen_experts, 0);
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

// rec-3cr: dump buffer for GEPA_DUMP builds (no-op storage otherwise). steps = local steps to capture.
static float* g_dump_dev = nullptr;
static size_t g_dump_n = 0;
extern "C" int mid_dump_setup(int steps) {
    size_t n = (size_t)steps * MID_NL * MID_DUMP_STRIDE;
    if (g_dump_dev) { hipFree(g_dump_dev); g_dump_dev = nullptr; }
    if (hipMalloc(&g_dump_dev, n * sizeof(float)) != hipSuccess) return -1;
    hipMemset(g_dump_dev, 0xff, n * sizeof(float));
    g_dump_n = n;
    if (hipMemcpyToSymbol(HIP_SYMBOL(g_mid_dump), &g_dump_dev, sizeof(float*)) != hipSuccess) return -2;
    if (hipMemcpyToSymbol(HIP_SYMBOL(g_mid_dump_steps), &steps, sizeof(int)) != hipSuccess) return -3;
    return GEPA_DUMP ? MID_DUMP_STRIDE : 0;
}
extern "C" int mid_dump_fetch(float* h, size_t n) {
    if (!g_dump_dev || n > g_dump_n) return -1;
    hipDeviceSynchronize();
    return hipMemcpy(h, g_dump_dev, n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess ? 0 : -2;
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
