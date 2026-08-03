#include <cfloat>
#include <stdio.h>




template<typename T, typename U, T (*Cmp)(T, T)>
__device__ __forceinline__ sri<T, U> warp_reduce_single_ind(sri<T, U> sr) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T down_red = __shfl_down_sync(0xffffffff, sr.red, offset);
        U down_ind = __shfl_down_sync(0xffffffff, sr.ind, offset);
        if (sr.red != Cmp(sr.red, down_red)) {
            sr.red = down_red;
            sr.ind = down_ind;
        } 
    }
    return sr;
}



template<typename T, typename U, T (*Map)(T), T (*Cmp)(T, T)>
__global__ void single_reduction_kernel(const T* a, sri<T, U>* blocksrs,
                                        int n, T identity, U ind_identity) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ sri<T, U> sdata[NUM_WARPS];

    sri<T, U> sr{identity, ind_identity};
    for (int k = 0; k < STRIDE; k++) {
        int currIdx = STRIDE * i + k;
        if (currIdx < n) {
            if (sr.red != Cmp(sr.red, a[currIdx])) {
                sr.red = a[currIdx]; 
                sr.ind = currIdx; 
            }
        } else {
            break;
        }
    }

    sri<T, U> warp_sr = warp_reduce_single_ind<T, U, Cmp>(sr);
    if (lane == 0) sdata[warp_id] = warp_sr;
    __syncthreads();

    if (tid == 0) {
        sri<T, U> final_sr{identity, ind_identity};
        for (int j = 0; j < NUM_WARPS; j++) {
            if (final_sr.red != Cmp(final_sr.red, sdata[j].red)) { 
                final_sr.red = sdata[j].red;
                final_sr.ind = sdata[j].ind;
            }
        }
        blocksrs[blockIdx.x] = final_sr;
    }
}


template<typename T, typename U, T (*Cmp)(T, T)>
__global__ void single_reduction_kernel_packed(const sri<T, U>* a,
                                               sri<T, U>* blocksrs,
                                               int n, T identity, U ind_identity) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ sri<T, U> sdata[NUM_WARPS];

    sri<T, U> sr{identity, ind_identity};
    for (int k = 0; k < STRIDE; k++) {
        int currIdx = STRIDE * i + k;
        if (currIdx < n) {
            if (sr.red != Cmp(sr.red, a[currIdx].red)) {
                sr.red = a[currIdx].red; 
                sr.ind = a[currIdx].ind; 
            }
        } else {
            break;
        }
    }

    sri<T, U> warp_sr = warp_reduce_single_ind<T, U, Cmp>(sr);
    if (lane == 0) sdata[warp_id] = warp_sr;
    __syncthreads();

    if (tid == 0) {
        sri<T, U> final_sr{identity, ind_identity};
        for (int j = 0; j < NUM_WARPS; j++) {
            if (final_sr.red != Cmp(final_sr.red, sdata[j].red)) {
                final_sr.red = sdata[j].red;
                final_sr.ind = sdata[j].ind;
            }
        }
        blocksrs[blockIdx.x] = final_sr;
    }
}


template<typename T, typename U, T (*Map)(T), T (*Cmp)(T, T)>
void single_reduction_launcher(const T* data, T* out0, U* out1, int n,
                              T identity, U ind_identity) {
    int current_n = n;
    int num_units = (current_n + STRIDE - 1) / STRIDE;
    int num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

    static thread_local sri<T, U>* device_output = nullptr;
    static thread_local size_t device_output_capacity = 0;

    if (num_blocks > device_output_capacity) {
        if (device_output) cudaFree(device_output);
        cudaMalloc(&device_output, num_blocks * sizeof(sri<T, U>));
        device_output_capacity = num_blocks;
    }

    single_reduction_kernel<T, U, Map, Cmp><<<num_blocks, BLOCK_SIZE>>>(
        data, device_output, current_n, identity, ind_identity);
    current_n = num_blocks;

    while (current_n > BLOCK_SIZE) {
        num_units = (current_n + STRIDE - 1) / STRIDE;
        num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

        single_reduction_kernel_packed<T, U, Cmp><<<num_blocks, BLOCK_SIZE>>>(
            device_output, device_output, current_n, identity, ind_identity);

        current_n = num_blocks;
    }

    single_reduction_kernel_packed<T, U, Cmp><<<1, BLOCK_SIZE>>>(
        device_output, device_output, current_n, identity, ind_identity);

    cudaMemcpy(out0, &device_output[0].red, sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out1, &device_output[0].ind, sizeof(U), cudaMemcpyDeviceToDevice);
}