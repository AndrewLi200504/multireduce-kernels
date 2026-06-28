#include <stdio.h>
#include <cfloat>

#include "reduction.h"


template<typename T> __device__ T dev_min(T a, T b) { return a < b ? a : b; }
template<typename T> __device__ T dev_max(T a, T b) { return a > b ? a : b; }
template<typename T> __device__ T dev_sum(T a, T b) { return a + b; }

template<typename T, T (*Op0)(T, T), T (*Op1)(T, T)>
__device__ __forceinline__ dual_reduction<T> warp_reduce_dual(dual_reduction<T> mm) {
    for (int offset = 16; offset > 0; offset /= 2) {
        mm.red0 = Op0(mm.red0, __shfl_down_sync(0xffffffff, mm.red0, offset));
        mm.red1 = Op1(mm.red1, __shfl_down_sync(0xffffffff, mm.red1, offset));
    }
    return mm;
}


template<typename T, T (*Op0)(T, T), T (*Op1)(T, T)>
__global__ void dual_reduction_kernel(const T* a, dual_reduction<T>* blockmms,
                                        int n, T identity0, T identity1) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ dual_reduction<T> sdata[NUM_WARPS];

    dual_reduction<T> mm;
    if (i < n) {
        mm.red0 = a[i];
        mm.red1 = a[i];
    } else {
        mm.red0 = identity0;   
        mm.red1 = identity1;
    }

    dual_reduction<T> warp_mm = warp_reduce_dual<T, Op0, Op1>(mm);
    if (lane == 0) sdata[warp_id] = warp_mm;
    __syncthreads();

    if (tid == 0) {
        dual_reduction<T> final_mm{identity0, identity1};
        for (int j = 0; j < NUM_WARPS; j++) {
            final_mm.red0 = Op0(final_mm.red0, sdata[j].red0);
            final_mm.red1 = Op1(final_mm.red1, sdata[j].red1);
        }
        blockmms[blockIdx.x] = final_mm;
    }
}

template<typename T, T (*Op0)(T, T), T (*Op1)(T, T)>
__global__ void dual_reduction_kernel_packed(const dual_reduction<T>* a,
                                               dual_reduction<T>* blockmms,
                                               int n, T identity0, T identity1) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ dual_reduction<T> sdata[NUM_WARPS];

    dual_reduction<T> mm;
    if (i < n) {
        mm.red0 = a[i].red0;
        mm.red1 = a[i].red1;
    } else {
        mm.red0 = identity0;
        mm.red1 = identity1;
    }

    dual_reduction<T> warp_mm = warp_reduce_dual<T, Op0, Op1>(mm);
    if (lane == 0) sdata[warp_id] = warp_mm;
    __syncthreads();

    if (tid == 0) {
        dual_reduction<T> final_mm{identity0, identity1};
        for (int j = 0; j < NUM_WARPS; j++) {
            final_mm.red0 = Op0(final_mm.red0, sdata[j].red0);
            final_mm.red1 = Op1(final_mm.red1, sdata[j].red1);
        }
        blockmms[blockIdx.x] = final_mm;
    }
}


template<typename T, T (*Op0)(T, T), T (*Op1)(T, T)>
void dual_reduction_launcher(const T* data, T* out0, T* out1, int n,
                              T identity0, T identity1) {
    int current_n = n;
    int num_blocks = (current_n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dual_reduction<T>* device_output;
    cudaMalloc(&device_output, num_blocks * sizeof(dual_reduction<T>));
    dual_reduction_kernel<T, Op0, Op1><<<num_blocks, BLOCK_SIZE>>>(
        data, device_output, current_n, identity0, identity1);

    dual_reduction<T>* intermediate = device_output;
    current_n = num_blocks;

    while (current_n > BLOCK_SIZE) {
        num_blocks = (current_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dual_reduction<T>* next_output;
        cudaMalloc(&next_output, num_blocks * sizeof(dual_reduction<T>));
        dual_reduction_kernel_packed<T, Op0, Op1><<<num_blocks, BLOCK_SIZE>>>(
            intermediate, next_output, current_n, identity0, identity1);
        cudaFree(intermediate);
        intermediate = next_output;
        current_n = num_blocks;
    }

    dual_reduction<T>* final_output;
    cudaMalloc(&final_output, sizeof(dual_reduction<T>));
    dual_reduction_kernel_packed<T, Op0, Op1><<<1, BLOCK_SIZE>>>(
        intermediate, final_output, current_n, identity0, identity1);
    cudaFree(intermediate);

    cudaMemcpy(out0, &final_output->red0, sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out1, &final_output->red1, sizeof(T), cudaMemcpyDeviceToDevice);
    cudaFree(final_output);
}

void min_max_launcher(float* data, float* min, float* max, int n) {
    dual_reduction_launcher<float, dev_min<float>, dev_max<float>>(
        data, min, max, n, FLT_MAX, -FLT_MAX);
}