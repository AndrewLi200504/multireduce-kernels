#include <cfloat>
#include <stdio.h>
#define BLOCK_SIZE 128
#define WARP_SIZE 32


template<typename T, typename U>
struct single_reduction_ind {
    T red;
    U ind; 
};

template<typename T, typename U>
using sri = single_reduction_ind<T, U>;

template<typename T>
__device__ __forceinline__ int dev_comp_min(T a, T b) {
    if (a == b) {
        return 0;
    } else if (a < b) {
        return 1;
    } else {
        return -1;
    }
}

template<typename T>
__device__ __forceinline__ int dev_comp_max(T a, T b) {
    if (a == b) {
        return 0;
    } else if (a > b) {
        return 1;
    } else {
        return -1;
    }
}
template<typename T, typename U, int (*Cmp)(T, T), int (*Cmpind)(U, U)>
__device__ __forceinline__ sri<T, U> warp_reduce_single_ind(sri<T, U> sr) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T down_red = __shfl_down_sync(0xffffffff, sr.red, offset);
        U down_ind = __shfl_down_sync(0xffffffff, sr.ind, offset);
        int cmp = Cmp(sr.red, down_red);
        if (cmp == 0) {
            sr.ind = Cmpind(sr.ind, down_ind) > 0 ? sr.ind : down_ind;
        } else {
            if (cmp < 0) {
                sr.red = down_red;
                sr.ind = down_ind;
            }
        }
    }
    return sr;
}



template<typename T, typename U, T (*Map)(T), int (*Cmp)(T, T), int (*Cmpind)(U, U)>
__global__ void single_reduction_kernel(const T* a, sri<T, U>* blocksrs,
                                        int n, T identity, U ind_identity) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ sri<T, U> sdata[NUM_WARPS];

    sri<T, U> sr;
    if (i < n) {
        sr.red = Map(a[i]);
        sr.ind = (U)i;
    } else {
        sr.red = identity;   
        sr.ind = ind_identity;
    }

    sri<T, U> warp_sr = warp_reduce_single_ind<T, U, Cmp, Cmpind>(sr);
    if (lane == 0) sdata[warp_id] = warp_sr;
    __syncthreads();

    if (tid == 0) {
        sri<T, U> final_sr{identity, ind_identity};
        for (int j = 0; j < NUM_WARPS; j++) {
            int cmp = Cmp(final_sr.red, sdata[j].red);
            if (cmp == 0) {
                final_sr.ind = Cmpind(sr.ind, sdata[j].ind) > 0 ? sr.ind : sdata[j].ind;
            } else {
                if (cmp < 0) {
                    final_sr.red = sdata[j].red;
                    final_sr.ind = sdata[j].ind;
                }
            }
        }
        blocksrs[blockIdx.x] = final_sr;
    }
}


template<typename T, typename U, int (*Cmp)(T, T), int (*Cmpind)(U, U)>
__global__ void single_reduction_kernel_packed(const sri<T, U>* a,
                                               sri<T, U>* blocksrs,
                                               int n, T identity, U ind_identity) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ sri<T, U> sdata[NUM_WARPS];

    sri<T, U> sr;
    if (i < n) {
        sr.red = a[i].red;
        sr.ind = a[i].ind;
    } else {
        sr.red = identity;
        sr.ind = ind_identity;
    }

    sri<T, U> warp_sr = warp_reduce_single_ind<T, U, Cmp, Cmpind>(sr);
    if (lane == 0) sdata[warp_id] = warp_sr;
    __syncthreads();

    if (tid == 0) {
        sri<T, U> final_sr{identity, ind_identity};
        for (int j = 0; j < NUM_WARPS; j++) {
            int cmp = Cmp(final_sr.red, sdata[j].red);
            if (cmp == 0) {
                final_sr.ind = Cmpind(sr.ind, sdata[j].ind) > 0 ? sr.ind : sdata[j].ind;
            } else {
                if (cmp < 0) {
                    final_sr.red = sdata[j].red;
                    final_sr.ind = sdata[j].ind;
                }
            }
        }
        blocksrs[blockIdx.x] = final_sr;
    }
}


template<typename T, typename U, T (*Map)(T), int (*Cmp)(T, T), int (*Cmpind)(U, U)>
void single_reduction_launcher(const T* data, T* out0, U* out1, int n,
                              T identity, U ind_identity) {
    int current_n = n;
    int num_blocks = (current_n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    static thread_local sri<T, U>* device_output = nullptr;
    static thread_local size_t device_output_capacity = 0;
    if (num_blocks > device_output_capacity) {
        if (device_output) cudaFree(device_output);
        cudaMalloc(&device_output, num_blocks * sizeof(sri<T, U>));
        device_output_capacity = num_blocks;
    }                            
    single_reduction_kernel<T, U, Map, Cmp, Cmpind><<<num_blocks, BLOCK_SIZE>>>(
        data, device_output, current_n, identity, ind_identity);

    current_n = num_blocks;

    while (current_n > BLOCK_SIZE) {
        num_blocks = (current_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sri<T, U>* next_output;
        single_reduction_kernel_packed<T, U, Cmp, Cmpind><<<num_blocks, BLOCK_SIZE>>>(
            device_output, device_output, current_n, identity, ind_identity);
        current_n = num_blocks;
    }

    single_reduction_kernel_packed<T, U, Cmp, Cmpind><<<1, BLOCK_SIZE>>>(
        device_output, device_output, current_n, identity, ind_identity);
    cudaMemcpy(out0, &device_output[0].red, sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out1, &device_output[0].ind, sizeof(U), cudaMemcpyDeviceToDevice);
}