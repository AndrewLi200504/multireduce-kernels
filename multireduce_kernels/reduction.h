#include <cfloat>
#include <stdio.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template<typename T>
struct dual_reduction {
    T red0;
    T red1;
};

template<typename T> __device__ T dev_nop(T a) { return a; }
template<typename T> __device__ T dev_sqr(T a) { return a * a; }
template<typename T> __device__ T dev_min(T a, T b) { return a < b ? a : b; }
template<typename T> __device__ T dev_max(T a, T b) { return a > b ? a : b; }
template<typename T> __device__ T dev_sum(T a, T b) { return a + b; }

template<typename T, T (*Op0)(T, T), T (*Op1)(T, T)>
__device__ __forceinline__ dual_reduction<T> warp_reduce_dual(dual_reduction<T> dr) {
    for (int offset = 16; offset > 0; offset /= 2) {
        dr.red0 = Op0(dr.red0, __shfl_down_sync(0xffffffff, dr.red0, offset));
        dr.red1 = Op1(dr.red1, __shfl_down_sync(0xffffffff, dr.red1, offset));
    }
    return dr;
}


template<typename T, T (*Map0) (T), T (*Map1) (T), T (*Op0)(T, T), T (*Op1)(T, T)>
__global__ void dual_reduction_kernel(const T* a, dual_reduction<T>* blockdrs,
                                        int n, T identity0, T identity1) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ dual_reduction<T> sdata[NUM_WARPS];

    dual_reduction<T> dr;
    if (i < n) {
        dr.red0 = Map0(a[i]);
        dr.red1 = Map1(a[i]);
    } else {
        dr.red0 = identity0;   
        dr.red1 = identity1;
    }

    dual_reduction<T> warp_dr = warp_reduce_dual<T, Op0, Op1>(dr);
    if (lane == 0) sdata[warp_id] = warp_dr;
    __syncthreads();

    if (tid == 0) {
        dual_reduction<T> final_dr{identity0, identity1};
        for (int j = 0; j < NUM_WARPS; j++) {
            final_dr.red0 = Op0(final_dr.red0, sdata[j].red0);
            final_dr.red1 = Op1(final_dr.red1, sdata[j].red1);
        }
        blockdrs[blockIdx.x] = final_dr;
    }
}

template<typename T, T (*Op0)(T, T), T (*Op1)(T, T)>
__global__ void dual_reduction_kernel_packed(const dual_reduction<T>* a,
                                               dual_reduction<T>* blockdrs,
                                               int n, T identity0, T identity1) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ dual_reduction<T> sdata[NUM_WARPS];

    dual_reduction<T> dr;
    if (i < n) {
        dr.red0 = a[i].red0;
        dr.red1 = a[i].red1;
    } else {
        dr.red0 = identity0;
        dr.red1 = identity1;
    }

    dual_reduction<T> warp_dr = warp_reduce_dual<T, Op0, Op1>(dr);
    if (lane == 0) sdata[warp_id] = warp_dr;
    __syncthreads();

    if (tid == 0) {
        dual_reduction<T> final_dr{identity0, identity1};
        for (int j = 0; j < NUM_WARPS; j++) {
            final_dr.red0 = Op0(final_dr.red0, sdata[j].red0);
            final_dr.red1 = Op1(final_dr.red1, sdata[j].red1);
        }
        blockdrs[blockIdx.x] = final_dr;
    }
}


template<typename T, T (*Map0) (T), T (*Map1) (T), T (*Op0)(T, T), T (*Op1)(T, T)>
void dual_reduction_launcher(const T* data, T* out0, T* out1, int n,
                              T identity0, T identity1) {
    int current_n = n;
    int num_blocks = (current_n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dual_reduction<T>* device_output;
    cudaMalloc(&device_output, num_blocks * sizeof(dual_reduction<T>));
    dual_reduction_kernel<T, Map0, Map1, Op0, Op1><<<num_blocks, BLOCK_SIZE>>>(
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
