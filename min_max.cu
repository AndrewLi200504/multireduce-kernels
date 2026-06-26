#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cfloat>
#include <time.h>
#include <cuda_runtime.h>
#include "min_max.h"

#define N 10000000  // Vector size = 10 million
#define BLOCK_SIZE 256
#define WARP_SIZE 32


min_max vector_min_max_cpu(float *a, int n) {
    min_max mm_cpu;
    for (int i = 0; i < n; i++) {
        if (a[i] < mm_cpu.min) {
            mm_cpu.min = a[i];
        }
        if (a[i] > mm_cpu.max) {
            mm_cpu.max = a[i];
        }
    }
    return mm_cpu;
}

__device__ __forceinline__ min_max warp_reduce_min_max(min_max mm) {
    for (int offset = 16; offset > 0; offset /= 2) {
        mm.min = fminf(mm.min, __shfl_down_sync(0xffffffff, mm.min, offset));
        mm.max = fmaxf(mm.max, __shfl_down_sync(0xffffffff, mm.max, offset));
    }
    return mm;
}

__global__ void vector_min_max_gpu(float *a, min_max* blockmms, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ min_max sdata[NUM_WARPS];

    __syncthreads();
    min_max mm;
    if (i < n) {
        mm.min = a[i];
        mm.max = a[i];
    }
    min_max warp_min_max = warp_reduce_min_max(mm);
    if (lane == 0) {
        sdata[warp_id] = warp_min_max;  
    }
    __syncthreads();  

    if (tid == 0) {
        min_max final_mm = SENTINEL;
        for (int j = 0; j < NUM_WARPS; j++) {
            min_max curr_mm = sdata[j];
            final_mm.min = fminf(final_mm.min, curr_mm.min);
            final_mm.max = fmaxf(final_mm.max, curr_mm.max);
        }
        blockmms[blockIdx.x] = final_mm;

    }
}


// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
int main() {
    float *host_input, *host_cpu;
    min_max *host_gpu;
    float *device_input;
    min_max *device_output;
    size_t size = N * sizeof(float);
    
    // Define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1025 + 256 - 1) / 256 ) = 1280 / 256 = 4 rounded 
    size_t output_size = num_blocks * sizeof(min_max);
    // Allocate host memory
    host_input = (float*)malloc(size);
    host_cpu = (float*)malloc(output_size);
    host_gpu = (min_max*)malloc(output_size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(host_input, N);

    // Allocate device memory
    cudaMalloc(&device_input, size);
    cudaMalloc(&device_output, output_size);

    // Copy data to device
    cudaMemcpy(device_input, host_input, size, cudaMemcpyHostToDevice);

    // Warm-up runs
    min_max min_max_cpu = vector_min_max_cpu(host_input, N);
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_min_max_cpu(host_input, N);
        vector_min_max_gpu<<<num_blocks, BLOCK_SIZE>>>(device_input, device_output, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_min_max_cpu(host_input, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    float max_result = -FLT_MAX;
    float min_result = FLT_MAX;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_min_max_gpu<<<num_blocks, BLOCK_SIZE>>>(device_input, device_output, N);
        cudaDeviceSynchronize();
        cudaMemcpy(host_gpu, device_output, output_size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_blocks; i++) {
            max_result = max(host_gpu[i].max, max_result); 
            min_result = min(host_gpu[i].min, min_result);
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
    if (fabs(min_max_cpu.max - max_result) > 1e-5) {
        correct = false;
    }
    if (fabs(min_max_cpu.min - min_result) > 1e-5) {
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
