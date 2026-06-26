#include <stdio.h>
#include <cfloat>

#include "max.h"

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) 
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ min_max warp_reduce_min_max(min_max mm) {
    for (int offset = 16; offset > 0; offset /= 2) {
        mm.min = fminf(mm.min, __shfl_down_sync(0xffffffff, mm.min, offset));
        mm.max = fmaxf(mm.max, __shfl_down_sync(0xffffffff, mm.max, offset));
    }
    return mm;
}
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


__global__ void vector_min_max_gpu(min_max *a, min_max* blockmms, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ min_max sdata[NUM_WARPS];

    __syncthreads();
    min_max mm;
    if (i < n) {
        mm.min = a[i].min;
        mm.max = a[i].max;
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

void max_launcher(float* data, float* scalar, int n) {
    int current_n = n;
    float* device_input = data;
    float *device_output;
    while (current_n > BLOCK_SIZE) {
        int num_blocks = (current_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc(&device_output, num_blocks * sizeof(float));
        vector_max_gpu<<<num_blocks, BLOCK_SIZE>>>(device_input, device_output, current_n);
        if (device_input != data) cudaFree(device_input);
        device_input = device_output;
        current_n = num_blocks;
    }    
    vector_max_gpu<<<1, BLOCK_SIZE>>>(device_input, scalar, current_n);
    if (device_input != data) cudaFree(device_input);
}

void min_max_launcher(float* data, float* min, float* max, int n) {
    int current_n = n;
    float* device_input = data;
    min_max* device_output;
    int num_blocks = (current_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&device_output, num_blocks * sizeof(min_max));
    vector_min_max_gpu<<<num_blocks, BLOCK_SIZE>>>(device_input, device_output, current_n);
    if (device_input != data) cudaFree(device_input);
    min_max* intermediate_input = device_output;
    current_n = num_blocks;
    while (current_n > BLOCK_SIZE) {
        int num_blocks = (current_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc(&device_output, num_blocks * sizeof(min_max));
        vector_min_max_gpu<<<num_blocks, BLOCK_SIZE>>>(intermediate_input, device_output, current_n);
        cudaFree(intermediate_input);
        intermediate_input = device_output;
        current_n = num_blocks;
    }    
    min_max* final_output;
    cudaMalloc(&final_output, sizeof(min_max));
    vector_min_max_gpu<<<1, BLOCK_SIZE>>>(intermediate_input, final_output, current_n);
    cudaMemcpy(min, &final_output->min, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(max, &final_output->max, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(final_output);
    if (device_input != data) cudaFree(device_input);
}