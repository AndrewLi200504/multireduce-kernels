
#include <cuda_runtime.h>
#include <cfloat>

struct min_max {
    float min = FLT_MAX;
    float max = -FLT_MAX;
};

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define SENTINEL {FLT_MAX, -FLT_MAX}
__device__ __forceinline__ min_max warp_reduce_min_max(min_max mm);
__global__ void vector_min_max_gpu(float *a, min_max* blockminmaxes, int n);