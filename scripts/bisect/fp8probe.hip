#include <hip/hip_runtime.h>
extern "C" __global__ void k(float* o, const int* a, const int* b, const float* x) {
    int t = threadIdx.x;
    o[t] = __builtin_amdgcn_dot4_f32_fp8_fp8(a[t], b[t], o[t]);                 // dot11
    int p = __builtin_amdgcn_cvt_pk_fp8_f32(x[t], x[t+1], 0, false);           // fp8 pack
    o[t+64] = __builtin_amdgcn_cvt_f32_fp8(p, 0);                               // fp8 unpack
    o[t+128] = (float)__builtin_amdgcn_sudot4(true, a[t], true, b[t], 0, false); // int8 dot4
}
