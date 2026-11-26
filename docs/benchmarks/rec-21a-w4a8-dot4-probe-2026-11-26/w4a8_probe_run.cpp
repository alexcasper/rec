// rec-21a step-2 probe runner (native C++ port of w4a8_probe_run.py — python CDLL launches are broken
// post-reboot on dusk: hipErrorInvalidKernelImage/invalid-device-function for every dlopen'd lib, while
// native processes launch fine; inf_amd owns that regression). dlopens w4a8_probe.so, checks correctness
// vs host references, times with hipEvents (min of 3 runs of REPS launches). Prints a JSON verdict block.
// Build: hipcc -O2 -std=c++20 w4a8_probe_run.cpp -o w4a8_probe_run -ldl
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#define K 2048
#define ROWB (K / 2)
#define NG64 (K / 64)   // 32
#define NG16 (K / 16)   // 128
#define NA4 (K / 4)     // 512

using LaunchFn = int (*)(const void*, const void*, const void*, const void*, void*, int);
using QFn = int (*)(const void*, void*, void*, int);

int main(int argc, char** argv) {
    const int rows = (argc > 1) ? atoi(argv[1]) : 4096;
    const int reps = (argc > 2) ? atoi(argv[2]) : 50;
    void* h = dlopen("./w4a8_probe.so", RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }
    auto p0 = (LaunchFn)dlsym(h, "p0_dequant_fma");
    using LaunchFnQ = int (*)(const void*, const void*, const void*, const void*, const void*, void*, int);
    auto p1 = (LaunchFnQ)dlsym(h, "p1_dot4_unpack");
    auto p3 = (QFn)dlsym(h, "p3_quantize");
    if (!p0 || !p1 || !p3) { printf("dlsym failed\n"); return 1; }

    std::mt19937_64 rng(2049);
    std::uniform_int_distribution<int> nib(0, 15);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);

    std::vector<uint8_t> w(rows * ROWB);
    for (auto& v : w) v = (uint8_t)nib(rng);
    std::vector<float> s32(rows * NG64), b32(rows * NG64);
    const bool degen = (argc > 3 && std::string(argv[3]) == "degen");
    for (auto& v : s32) v = ud(rng);
    for (auto& v : b32) v = (ud(rng) - 1.0f) * 0.02f;
    if (degen) { std::fill(s32.begin(), s32.end(), 1.0f); std::fill(b32.begin(), b32.end(), 0.0f); }
    auto to_bf16 = [](float v) { uint32_t u; __builtin_memcpy(&u, &v, 4); return (uint16_t)(u >> 16); };
    auto from_bf16 = [](uint16_t h16) { uint32_t u = (uint32_t)h16 << 16; float f; __builtin_memcpy(&f, &u, 4); return (double)f; };
    std::vector<uint16_t> s_bf(rows * NG64), b_bf(rows * NG64);
    for (size_t i = 0; i < s32.size(); ++i) { s_bf[i] = to_bf16(s32[i]); b_bf[i] = to_bf16(b32[i]); }
    std::vector<float> x(K);
    for (auto& v : x) v = nd(rng);

    // host references (float64) — g64-affine: w_deq[k] = sv[k/64]*nib[k] + bv[k/64]
    std::vector<double> wq(rows * (size_t)K);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < ROWB; ++j) {
            const uint8_t byte = w[(size_t)r * ROWB + j];
            wq[(size_t)r * K + 2 * j] = double(byte & 0xF);
            wq[(size_t)r * K + 2 * j + 1] = double(byte >> 4);
        }
    std::vector<double> sve(rows * (size_t)K), bve(rows * (size_t)K);
    for (int r = 0; r < rows; ++r)
        for (int g = 0; g < NG64; ++g) {
            const double sv = from_bf16(s_bf[(size_t)r * NG64 + g]), bv = from_bf16(b_bf[(size_t)r * NG64 + g]);
            for (int t = 0; t < 64; ++t) { sve[(size_t)r * K + g * 64 + t] = sv; bve[(size_t)r * K + g * 64 + t] = bv; }
        }
    std::vector<double> yref(rows);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += (sve[(size_t)r * K + k] * wq[(size_t)r * K + k] + bve[(size_t)r * K + k]) * double(x[k]);
        yref[r] = acc;
    }

    // host quantizer (matches P3): per-g16 absmax/RNE -> s8 + scales, packed A-dwords
    std::vector<int8_t> qa(K);
    std::vector<float> sa(NG16);
    for (int g16 = 0; g16 < NG16; ++g16) {
        float m = 0.0f;
        for (int j = 0; j < 16; ++j) m = fmaxf(m, fabsf(x[g16 * 16 + j]));
        const float sc = fmaxf(m, 1e-30f) / 127.0f;
        for (int j = 0; j < 16; ++j)
            qa[g16 * 16 + j] = (int8_t)std::clamp(std::rint(x[g16 * 16 + j] / sc), -127.0f, 127.0f);
        sa[g16] = sc;
    }
    std::vector<int> qa4(NA4);
    for (int i = 0; i < NA4; ++i)
        for (int t = 0; t < 4; ++t) qa4[i] |= int(uint8_t(qa[i * 4 + t])) << (8 * t);
    std::vector<double> xq(K);
    for (int g16 = 0; g16 < NG16; ++g16)
        for (int j = 0; j < 16; ++j) xq[g16 * 16 + j] = double(sa[g16]) * double(qa[g16 * 16 + j]);
    std::vector<double> yrefq(rows);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += (sve[(size_t)r * K + k] * wq[(size_t)r * K + k] + bve[(size_t)r * K + k]) * xq[k];
        yrefq[r] = acc;
    }

    // device buffers
    void *dw, *ds, *db, *dx, *dy, *dqa4, *dsa;
    hipMalloc(&dw, w.size()); hipMalloc(&ds, s_bf.size() * 2); hipMalloc(&db, b_bf.size() * 2);
    hipMalloc(&dx, x.size() * 4); hipMalloc(&dy, size_t(rows) * 4);
    hipMalloc(&dqa4, qa4.size() * 4); hipMalloc(&dsa, sa.size() * 4);
    hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice);
    hipMemcpy(ds, s_bf.data(), s_bf.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(db, b_bf.data(), b_bf.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dqa4, qa4.data(), qa4.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dsa, sa.data(), sa.size() * 4, hipMemcpyHostToDevice);

    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    auto timed = [&](auto fn) {
        float best = 1e30f;
        for (int run = 0; run < 3; ++run) {
            hipEventRecord(e0);
            for (int i = 0; i < reps; ++i) fn();
            hipEventRecord(e1);
            hipEventSynchronize(e1);
            float ms; hipEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;  // ms for REPS launches
    };
    std::vector<float> ybuf(rows);
    auto relerr = [&](const std::vector<double>& ref, int topn) {
        hipMemcpy(ybuf.data(), dy, ybuf.size() * 4, hipMemcpyDeviceToHost);
        double mx = 0.0, nm = 0.0;
        std::vector<std::pair<double, int>> errs;
        for (int r = 0; r < rows; ++r) {
            const double e = std::fabs(double(ybuf[r]) - ref[r]);
            mx = std::max(mx, e); nm = std::max(nm, std::fabs(ref[r]));
            errs.push_back({e, r});
        }
        std::sort(errs.begin(), errs.end(), std::greater<>());
        for (int i = 0; i < topn && i < (int)errs.size(); ++i)
            printf("  worst row %d: dev=%.6f ref=%.6f err=%.6f\n", errs[i].second, ybuf[errs[i].second], ref[errs[i].second], errs[i].first);
        return nm > 0 ? mx / nm : mx;
    };

    int rc0 = p0(dw, ds, db, dx, dy, rows); hipDeviceSynchronize();
    float t0 = timed([&] { p0(dw, ds, db, dx, dy, rows); });
    double r0 = relerr(yref, 0);
    int rc1 = p1(dw, ds, db, dqa4, dsa, dy, rows); hipDeviceSynchronize();
    float t1 = timed([&] { p1(dw, ds, db, dqa4, dsa, dy, rows); });
    double r1 = relerr(yrefq, degen ? 6 : 0);

    const int nvec = 512;
    std::vector<float> xv(size_t(nvec) * K);
    for (auto& v : xv) v = nd(rng);
    void *dxv, *dqa4v, *dsav;
    hipMalloc(&dxv, xv.size() * 4); hipMalloc(&dqa4v, size_t(nvec) * NA4 * 4); hipMalloc(&dsav, size_t(nvec) * NG16 * 4);
    hipMemcpy(dxv, xv.data(), xv.size() * 4, hipMemcpyHostToDevice);
    int rc3 = p3(dxv, dqa4v, dsav, nvec); hipDeviceSynchronize();
    float t3 = timed([&] { p3(dxv, dqa4v, dsav, nvec); });
    std::vector<uint32_t> qback(size_t(nvec) * NA4), dummy;
    std::vector<float> sback(size_t(nvec) * NG16);
    hipMemcpy(qback.data(), dqa4v, qback.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(sback.data(), dsav, sback.size() * 4, hipMemcpyDeviceToHost);
    double r3 = 0.0;
    for (int v = 0; v < nvec; ++v)
        for (int g16 = 0; g16 < NG16; ++g16) {
            const float sc = sback[(size_t)v * NG16 + g16];
            for (int j = 0; j < 16; ++j) {
                const int8_t q = int8_t((qback[(size_t)v * NA4 + g16 * 4 + (j >> 2)] >> (8 * (j & 3))) & 0xFF);
                r3 = std::max(r3, std::fabs(double(sc) * double(q) - double(xv[(size_t)v * K + g16 * 16 + j])));
            }
        }

    printf("{\n");
    printf(" \"rows\": %d, \"reps\": %d, \"K\": %d,\n", rows, reps, K);
    printf(" \"p0_dequant_fma\": {\"ms_per_launch\": %.4f, \"rel_err_max\": %.3e, \"us_per_gemv\": %.2f, \"rc\": %d},\n",
           t0 / reps, r0, t0 / reps * 1e3, rc0);
    printf(" \"p1_dot4_unpack\": {\"ms_per_launch\": %.4f, \"rel_err_max\": %.3e, \"us_per_gemv\": %.2f, \"speedup_vs_p0\": %.3f, \"rc\": %d},\n",
           t1 / reps, r1, t1 / reps * 1e3, t0 / t1, rc1);
    printf(" \"p3_quantize\": {\"ms_per_512_vecs\": %.4f, \"us_per_vec\": %.2f, \"max_dequant_abs_err\": %.3e, \"rc\": %d},\n",
           t3 / reps, t3 / reps / nvec * 1e3, r3, rc3);
    printf(" \"note\": \"P1 conservative scalar nibble unpack; A-dwords pre-packed by the quantize pass.\"\n}\n");
    return 0;
}
