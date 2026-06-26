#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cfloat>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 10000000  // Vector size = 10 million
#define BLOCK_SIZE 256
#define WARP_SIZE 32
// Example:
// A = [1, 2, 3, 4, 5]
// vector_max(A) = 5

// CPU vector max
float vector_max_cpu(float *a, int n) {
    float max = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (a[i] > max) {
            max = a[i];
        }
    }
    return max;
}


__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) 
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
// CUDA kernel for vector max
__global__ void vector_max_gpu(float *a, float* blockmaxes, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ float sdata[NUM_WARPS];

    if (tid < NUM_WARPS) sdata[tid] = -FLT_MAX;
    __syncthreads();
    float val = (i < n) ? a[i] : -FLT_MAX;   
    float warp_max = warp_reduce_max(val);
    if (lane == 0) {
        sdata[warp_id] = warp_max;  
    }
    __syncthreads();  

    if (tid == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j < NUM_WARPS; j++) {
            m = fmaxf(m, sdata[j]);
        }
        blockmaxes[blockIdx.x] = m;

    }
}


// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *host_input, *host_cpu, *host_gpu;
    float *device_input, *device_output;
    size_t size = N * sizeof(float);
    
    // Define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1025 + 256 - 1) / 256 ) = 1280 / 256 = 4 rounded 
    size_t output_size = num_blocks * sizeof(float);
    // Allocate host memory
    host_input = (float*)malloc(size);
    host_cpu = (float*)malloc(output_size);
    host_gpu = (float*)malloc(output_size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(host_input, N);

    // Allocate device memory
    cudaMalloc(&device_input, size);
    cudaMalloc(&device_output, output_size);

    // Copy data to device
    cudaMemcpy(device_input, host_input, size, cudaMemcpyHostToDevice);

    // Warm-up runs
    float max_cpu = vector_max_cpu(host_input, N);
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_max_cpu(host_input, N);
        vector_max_gpu<<<num_blocks, BLOCK_SIZE>>>(device_input, device_output, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_max_cpu(host_input, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    float result = -FLT_MAX;

    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_max_gpu<<<num_blocks, BLOCK_SIZE>>>(device_input, device_output, N);
        cudaDeviceSynchronize();
        cudaMemcpy(host_gpu, device_output, output_size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_blocks; i++) {
            result = max(host_gpu[i], result); 
        }
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Verify results (optional)
    
    bool correct = true;
    if (fabs(max_cpu - result) > 1e-5) {
        correct = false;
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free memory
    free(host_input);
    free(host_cpu);
    free(host_gpu);
    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}
