#include <cuda_runtime.h>

extern "C" __global__ void null_stability(float* values, const float margin, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = values[idx];
        values[idx] = v > margin ? margin : v;
    }
}
