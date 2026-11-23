// rec-11b stage B: sunny-2gg-style encoding probes for the class-B VMEM hazard on gfx1150.
// Class B = VMEM result-register reuse (WAW/WAR) with waitcnt elision + dead XNACK-gated
// clause guard (ops_sunny memo 6dbb2bc / rec3cr-k4096-row0-divergence-isa-answer.md).
// Four probes, each run NREPS times; any output mismatch vs the CPU reference counts.
//   P0 control:        sequential loads to distinct destinations (RAW waits always emitted)
//   P1 same-type WAW:  two global loads into ONE destination reg (in-order WAW elision, legal)
//   P2 call-boundary:  caller hoists a loaded value across a noinline callee that forces
//                      scratch spills (launch_bounds occupancy cap) -> callee loads/stores
//                      share the caller's destination VGPR range; no explicit waits anywhere
//   P3 shrink replica: the q4_rows_tail structure (noinline tail fn + staged caller), tiny dims
// All launches are microseconds; every launch < 0.25 s (dusk display-GPU rule).
// Build: hipcc -O2 -shared -fPIC --offload-arch=gfx1150 fut11b_probes.hip -o fut11b_probes.so
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define NREPS 2000
#define N 4096  // rows per probe buffer (each row handled by lane 0 of its quad conceptually)

__device__ __attribute__((noinline)) float p2_callee(const float* __restrict__ spill_src,
                                                     const float* __restrict__ ld_src,
                                                     float keep, int i, int scramble) {
    // Force a large live set so RA spills to scratch under the occupancy cap, with
    // scratch_store -> scratch_load round trips the shrink-class says never get vscnt waits.
    float acc[48];
    #pragma unroll
    for (int k = 0; k < 48; ++k)
        acc[k] = spill_src[(i * 48 + k) & (N * 48 - 1)] * (1.0f + (float)k);
    float s = keep;
    #pragma unroll 4
    for (int k = 0; k < 48; ++k)
        s += acc[(k + scramble) % 48] * 1e-9f;          // keep all 48 live across traffic
    float l = ld_src[i];                                 // callee-side load, dest likely in caller's range
    return s * 0.5f + l * 0.5f;
}

// P3: replica of the gemv tail structure: noinline callee, block loop, caller staged path.
__device__ __attribute__((noinline)) float p3_tail(const float* __restrict__ w,
                                                   const float* __restrict__ x,
                                                   int row, int K) {
    float acc = 0.0f;
    #pragma unroll 1
    for (int b = 0; b < K; b += 64) {
        float v[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) v[k] = x[b + k];
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) sum += v[k];
        float d = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) d += w[row * K + b + k] * v[k];
        acc += d + sum * 1e-9f;
    }
    return acc;
}

__global__ void fut11b_probes(const float* __restrict__ A, const float* __restrict__ B,
                              const float* __restrict__ C, float* __restrict__ out,
                              int* __restrict__ mism, int probe) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float r;
    switch (probe) {
    case 0:  // P0 control: distinct destinations, sequential RAW-consumed loads
    {
        float t0 = A[i];
        float u0 = t0 * 2.0f + 1.0f;
        float t1 = B[i];
        float u1 = t1 * 2.0f + 1.0f;
        r = u0 + u1;
        break;
    }
    case 1:  // P1 same-type WAW: two loads, one destination, first value live only on an untaken path
    {
        float t = A[i];
        float u = 0.0f;
        if (t < -1e30f) { u = t; }      // never true at runtime; keeps first load destination-live
        t = B[i];
        r = t + u;
        break;
    }
    case 2:  // P2 call-boundary: caller value hoisted across a spilling noinline callee
    {
        float keep = A[i];              // caller-side load, live across the call
        r = p2_callee(C, B, keep, i, (int)keep & 47) + keep * 1e-9f;
        break;
    }
    default:  // P3 shrink replica
    {
        float a = 0.0f;
        #pragma unroll 1
        for (int b = 0; b < 64; b += 64) { a += A[i + b]; }
        r = p3_tail(C, A, i & 1023, 64) + a * 1e-9f;
        break;
    }
    }
    out[i] = r;
}

static void cpu_ref(int probe, const std::vector<float>& A, const std::vector<float>& B,
                    const std::vector<float>& C, std::vector<float>& ref) {
    for (int i = 0; i < N; ++i) {
        float r;
        switch (probe) {
        case 0: r = (A[i] * 2 + 1) + (B[i] * 2 + 1); break;
        case 1: { float u = (A[i] < -1e30f) ? A[i] : 0.0f; r = B[i] + u; break; }
        case 2: {
            float keep = A[i], s = keep;
            float acc[48];
            for (int k = 0; k < 48; ++k) acc[k] = C[(i * 48 + k) & (N * 48 - 1)] * (1.0f + (float)k);
            int sc = (int)keep & 47;
            for (int k = 0; k < 48; ++k) s += acc[(k + sc) % 48] * 1e-9f;
            r = s * 0.5f + B[i] * 0.5f + keep * 1e-9f;
            break;
        }
        default: {
            float a = A[i] + A[i + 64];
            float acc = 0.0f; int row = i & 1023;
            for (int b = 0; b < 64; b += 64) {
                float v[16], sum = 0.0f, d = 0.0f;
                for (int k = 0; k < 16; ++k) v[k] = A[b + k];
                for (int k = 0; k < 16; ++k) sum += v[k];
                for (int k = 0; k < 16; ++k) d += C[row * 64 + b + k] * v[k];
                acc += d + sum * 1e-9f;
            }
            r = acc + a * 1e-9f;
            break;
        }
        }
        ref[i] = r;
    }
}

int main() {
    hipError_t e = hipInit(0);
    if (e != hipSuccess) { printf("{\"status\":\"hip_init_failed\"}\n"); return 1; }
    printf("{\"status\":\"start\",\"nreps\":%d,\"n\":%d}\n", NREPS, N);
    std::vector<float> hA(N + 64), hB(N), hC(N * 64 + 1024);
    srand(1101);
    for (auto& v : hA) v = (float)(rand() % 2000 - 1000) / 8.0f;
    for (auto& v : hB) v = (float)(rand() % 2000 - 1000) / 8.0f;
    for (auto& v : hC) v = (float)(rand() % 2000 - 1000) / 8.0f;
    float *A, *B, *C, *out;
    int* mism;
    hipMalloc(&A, (N + 64) * 4); hipMalloc(&B, N * 4); hipMalloc(&C, hC.size() * 4);
    hipMalloc(&out, N * 4); hipMalloc(&mism, 4);
    hipMemcpy(A, hA.data(), (N + 64) * 4, hipMemcpyHostToDevice);
    hipMemcpy(B, hB.data(), N * 4, hipMemcpyHostToDevice);
    hipMemcpy(C, hC.data(), hC.size() * 4, hipMemcpyHostToDevice);
    int total_bad = 0;
    double worst = 0.0;
    for (int probe = 0; probe < 4; ++probe) {
        std::vector<float> ref(N);
        cpu_ref(probe, hA, hB, hC, ref);
        int bad = 0; double w = 0.0;
        for (int rep = 0; rep < NREPS; ++rep) {
            hipMemsetAsync(out, 0, N * 4, 0);
            hipLaunchKernelGGL(fut11b_probes, dim3((N + 63) / 64), dim3(64), 0, 0,
                               A, B, C, out, mism, probe);
            std::vector<float> h(N);
            hipMemcpy(h.data(), out, N * 4, hipMemcpyDeviceToHost);
            for (int i = 0; i < N; ++i) {
                double d = fabs((double)h[i] - (double)ref[i]);
                if (!(d <= 2e-3 * (1.0 + fabs((double)ref[i])))) {
                    ++bad;
                    if (d > w) w = d;
                }
            }
            hipStreamSynchronize(0);
        }
        printf("{\"probe\":%d,\"bad_reps_elements\":%d,\"worst_abs_dev\":%.6g}\n", probe, bad, w);
        total_bad += bad;
        if (w > worst) worst = w;
    }
    printf("{\"status\":\"done\",\"total_bad\":%d,\"worst\":%.6g,\"verdict\":\"%s\"}\n",
           total_bad, worst, total_bad == 0 ? "CLEAN" : "HAZARD_EXPOSED");
    return 0;
}
