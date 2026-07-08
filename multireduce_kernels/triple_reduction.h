#include <cfloat>
#include <stdio.h>
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define STRIDE 8

template<typename T> 
struct triple_reduction {
    T red0;
    T red1;
    T red2;
};

template<typename T> __device__ __forceinline__ T dev_mult(T a, T b) { return a * b; }

template<typename T, T (*Op0)(T, T), T (*Op1)(T, T), T (*Op2)(T, T)>
__device__ __forceinline__ triple_reduction<T> warp_reduce_triple(triple_reduction<T> tr) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        tr.red0 = Op0(tr.red0, __shfl_down_sync(0xffffffff, tr.red0, offset));
        tr.red1 = Op1(tr.red1, __shfl_down_sync(0xffffffff, tr.red1, offset));
        tr.red2 = Op2(tr.red2, __shfl_down_sync(0xffffffff, tr.red2, offset));
    }
    return tr;
}


template<typename T, T (*Map0)(T), T (*Map1)(T, T), T (*Map2)(T), T (*Op0)(T, T), T (*Op1)(T, T), T (*Op2)(T, T)>
__global__ void triple_reduction_kernel(const T* a, const T* b, triple_reduction<T>* blocktrs,
                                        int n, T identity0, T identity1) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ triple_reduction<T> sdata[NUM_WARPS];

    triple_reduction<T> tr{identity0, Map1(identity0, identity1), identity1};
    for (int k = 0; k < STRIDE; k++) {
        int currIdx = STRIDE * i + k;
        if (currIdx < n) {
            tr.red0 = Op0(tr.red0, Map0(a[currIdx]));
            tr.red1 = Op1(tr.red1, Map1(a[currIdx], b[currIdx])); 
            tr.red2 = Op2(tr.red2, Map2(b[currIdx]));
        } else {
            tr.red0 = Op0(tr.red0, Map0(identity0));
            tr.red1 = Op1(tr.red1, Map1(identity0, identity1));
            tr.red2 = Op2(tr.red2, Map2(identity1));
        }
    }

    triple_reduction<T> warp_tr = warp_reduce_triple<T, Op0, Op1, Op2>(tr);
    if (lane == 0) sdata[warp_id] = warp_tr;
    __syncthreads();

    if (tid == 0) {
        triple_reduction<T> final_tr{identity0, Map1(identity0, identity1), identity1};
        for (int j = 0; j < NUM_WARPS; j++) {
            final_tr.red0 = Op0(final_tr.red0, sdata[j].red0);
            final_tr.red1 = Op1(final_tr.red1, sdata[j].red1);
            final_tr.red2 = Op2(final_tr.red2, sdata[j].red2);
        }
        blocktrs[blockIdx.x] = final_tr;
    }
}

template<typename T, T (*Map1)(T, T), T(*Op0)(T, T), T(*Op1)(T, T), T (*Op2)(T, T)>
__global__ void triple_reduction_kernel_packed(const triple_reduction<T>* a,
                                               triple_reduction<T>* blocktrs,
                                               int n, T identity0, T identity1) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ triple_reduction<T> sdata[NUM_WARPS];

    triple_reduction<T> tr{identity0, Map1(identity0, identity1), identity1};
    for (int k = 0; k < STRIDE; k++) {
        int currIdx = STRIDE * i + k;
        if (currIdx < n) {
            tr.red0 = Op0(tr.red0, a[currIdx].red0);
            tr.red1 = Op1(tr.red1, a[currIdx].red1);
            tr.red2 = Op2(tr.red2, a[currIdx].red2);
        } else {
            tr.red0 = Op0(tr.red0, identity0);
            tr.red1 = Op1(tr.red1, Map1(identity0, identity1));
            tr.red2 = Op2(tr.red2, identity1);
        }
    }
    triple_reduction<T> warp_tr = warp_reduce_triple<T, Op0, Op1, Op2>(tr);
    if (lane == 0) sdata[warp_id] = warp_tr;
    __syncthreads();

    if (tid == 0) {
        triple_reduction<T> final_tr{identity0, Map1(identity0, identity1), identity1};
        for (int j = 0; j < NUM_WARPS; j++) {
            final_tr.red0 = Op0(final_tr.red0, sdata[j].red0);
            final_tr.red1 = Op1(final_tr.red1, sdata[j].red1);
            final_tr.red2 = Op2(final_tr.red2, sdata[j].red2);
        }
        blocktrs[blockIdx.x] = final_tr;
    }
}

template<typename T, T (*Map0)(T), T (*Map1)(T, T), T (*Map2)(T), 
T (*Op0)(T, T), T (*Op1)(T, T), T (*Op2)(T, T)>
void triple_reduction_launcher(const T* data0, const T* data1, T* out0, T* out1, T* out2, int n,
                              T identity0, T identity1) {
    int current_n = n;
    int num_units = (current_n + STRIDE - 1) / STRIDE;
    int num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

    static thread_local triple_reduction<T>* device_output = nullptr;
    static thread_local size_t device_output_capacity = 0;

    if (num_blocks > device_output_capacity) {
        if (device_output) cudaFree(device_output);
        cudaMalloc(&device_output, num_blocks * sizeof(triple_reduction<T>));
        device_output_capacity = num_blocks;
    }

    triple_reduction_kernel<T, Map0, Map1, Map2, Op0, Op1, Op2><<<num_blocks, BLOCK_SIZE>>>(
        data0, data1, device_output, current_n, identity0, identity1);
    current_n = num_blocks;

    while (current_n > BLOCK_SIZE) {
        num_units = (current_n + STRIDE - 1) / STRIDE;
        num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

        triple_reduction_kernel_packed<T, Map1, Op0, Op1, Op2><<<num_blocks, BLOCK_SIZE>>>(
            device_output, device_output, current_n, identity0, identity1);

        current_n = num_blocks;
    }

    triple_reduction_kernel_packed<T, Map1, Op0, Op1, Op2><<<1, BLOCK_SIZE>>>(
        device_output, device_output, current_n, identity0, identity1);

    cudaMemcpy(out0, &device_output[0].red0, sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out1, &device_output[0].red1, sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out2, &device_output[0].red2, sizeof(T), cudaMemcpyDeviceToDevice);
}