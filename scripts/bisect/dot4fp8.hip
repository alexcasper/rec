#include <hip/hip_runtime.h>
extern "C" __global__ void k8(float* of, const int* x, const int* y) {
    int t = threadIdx.x;
    of[t] = __builtin_amdgcn_dot4_f32_bf8_bf8(x[t], y[t], of[t]);
}
