#include <hip/hip_runtime.h>
extern "C" __global__ void k4(const int* a, const int* b, int* o, float* of, const unsigned* x, const unsigned* y) {
    int t = threadIdx.x;
    o[t] = __builtin_amdgcn_sudot4(true, a[t], true, b[t], o[t], false);   // v_dot4_i32_iu8 (RDNA3)
#if __has_builtin(__builtin_amdgcn_dot4_f32_bf8_bf8)
    of[t] = 1.0f;
#endif
}
