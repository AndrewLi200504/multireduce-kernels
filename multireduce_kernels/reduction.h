#include <cfloat>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template<typename T>
struct dual_reduction {
    T red0;
    T red1;
};


__global__ void vector_min_max_gpu(float *a, float* blockminmaxes, int n);