#include <cfloat>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define SENTINEL {FLT_MAX, -FLT_MAX}

struct min_max {
    float min = FLT_MAX;
    float max = -FLT_MAX;
};

__global__ void vector_max_gpu(float *a, float* blockmaxes, int n);
__global__ void vector_min_max_gpu(float *a, float* blockminmaxes, int n);