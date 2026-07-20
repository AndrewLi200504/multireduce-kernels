#include <cfloat>
#include <stdio.h>
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define STRIDE 8

template<typename T> 
struct quad_reduction {
    T red0;
    T red1;
    T red2;
    T red3;
};


template<typename T, T (*Op0)(T, T), T (*Op1)(T, T), T (*Op2)(T, T), T (*Op3)(T, T)>
__device__ __forceinline__ quad_reduction<T> warp_reduce_quad(quad_reduction<T>qr) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
       qr.red0 = Op0(qr.red0, __shfl_down_sync(0xffffffff,qr.red0, offset));
       qr.red1 = Op1(qr.red1, __shfl_down_sync(0xffffffff,qr.red1, offset));
       qr.red2 = Op2(qr.red2, __shfl_down_sync(0xffffffff,qr.red2, offset));
       qr.red3 = Op3(qr.red3, __shfl_down_sync(0xffffffff,qr.red3, offset));
    }
    return qr;
}


template<typename T, T (*Map0)(T), T (*Map1)(T, T), T (*Map2)(T), T (*Map3)(T), T (*Op0)(T, T), T (*Op1)(T, T), T (*Op2)(T, T), T (*Op3) (T, T)>
__global__ void quad_reduction_kernel(const T* a, const T* b, quad_reduction<T>* blockqrs,
                                        int n, T identity0, T identity1, T identity2) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ quad_reduction<T> sdata[NUM_WARPS];

    quad_reduction<T> qr{identity0, Map1(identity0, identity1), identity1, identity2};
    for (int k = 0; k < STRIDE; k++) {
        int currIdx = STRIDE * i + k;
        if (currIdx < n) {
            qr.red0 = Op0(qr.red0, Map0(a[currIdx]));
            qr.red1 = Op1(qr.red1, Map1(a[currIdx], b[currIdx])); 
            qr.red2 = Op2(qr.red2, Map2(b[currIdx]));
            qr.red3 = Op3(qr.red3, Map3(a[currIdx]));
        } else {
            qr.red0 = Op0(qr.red0, Map0(identity0));
            qr.red1 = Op1(qr.red1, Map1(identity0, identity1));
            qr.red2 = Op2(qr.red2, Map2(identity1));
            qr.red3 = Op3(qr.red3, Map3(identity2));
        }
    }

    quad_reduction<T> warp_qr = warp_reduce_quad<T, Op0, Op1, Op2, Op3>(qr);
    if (lane == 0) sdata[warp_id] = warp_qr;
    __syncthreads();

    if (tid == 0) {
        quad_reduction<T> final_qr{identity0, Map1(identity0, identity1), identity1, identity2};
        for (int j = 0; j < NUM_WARPS; j++) {
            final_qr.red0 = Op0(final_qr.red0, sdata[j].red0);
            final_qr.red1 = Op1(final_qr.red1, sdata[j].red1);
            final_qr.red2 = Op2(final_qr.red2, sdata[j].red2);
            final_qr.red3 = Op3(final_qr.red3, sdata[j].red3);

        }
        blockqrs[blockIdx.x] = final_qr;
    }
}

template<typename T, T (*Map1)(T, T), T(*Op0)(T, T), T(*Op1)(T, T), T (*Op2)(T, T), T (*Op3)(T, T)>
__global__ void quad_reduction_kernel_packed(const quad_reduction<T>* a,
                                               quad_reduction<T>* blockqrs,
                                               int n, T identity0, T identity1, T identity2) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ quad_reduction<T> sdata[NUM_WARPS];

    quad_reduction<T> qr{identity0, Map1(identity0, identity1), identity1, identity2};
    for (int k = 0; k < STRIDE; k++) {
        int currIdx = STRIDE * i + k;
        if (currIdx < n) {
            qr.red0 = Op0(qr.red0, a[currIdx].red0);
            qr.red1 = Op1(qr.red1, a[currIdx].red1);
            qr.red2 = Op2(qr.red2, a[currIdx].red2);
            qr.red3 = Op3(qr.red3, a[currIdx].red3);

        } else {
            qr.red0 = Op0(qr.red0, identity0);
            qr.red1 = Op1(qr.red1, Map1(identity0, identity1));
            qr.red2 = Op2(qr.red2, identity1);
            qr.red3 = Op3(qr.red3, identity2);
        }
    }
    quad_reduction<T> warp_qr = warp_reduce_quad<T, Op0, Op1, Op2, Op3>(qr);
    if (lane == 0) sdata[warp_id] = warp_qr;
    __syncthreads();

    if (tid == 0) {
        quad_reduction<T> final_qr{identity0, Map1(identity0, identity1), identity1, identity2};
        for (int j = 0; j < NUM_WARPS; j++) {
            final_qr.red0 = Op0(final_qr.red0, sdata[j].red0);
            final_qr.red1 = Op1(final_qr.red1, sdata[j].red1);
            final_qr.red2 = Op2(final_qr.red2, sdata[j].red2);
            final_qr.red3 = Op3(final_qr.red3, sdata[j].red3);
        }
        blockqrs[blockIdx.x] = final_qr;
    }
}

template<typename T, T (*Map0)(T), T (*Map1)(T, T), T (*Map2)(T), T (*Map3)(T),
T (*Op0)(T, T), T (*Op1)(T, T), T (*Op2)(T, T), T (*Op3)(T, T)>
void quad_reduction_launcher(const T* data0, const T* data1, T* out0, T* out1, T* out2, T* out3, int n,
                              T identity0, T identity1, T identity2) {
    int current_n = n;
    int num_units = (current_n + STRIDE - 1) / STRIDE;
    int num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

    static thread_local quad_reduction<T>* device_output = nullptr;
    static thread_local size_t device_output_capacity = 0;

    if (num_blocks > device_output_capacity) {
        if (device_output) cudaFree(device_output);
        cudaMalloc(&device_output, num_blocks * sizeof(quad_reduction<T>));
        device_output_capacity = num_blocks;
    }

    quad_reduction_kernel<T, Map0, Map1, Map2, Map3, Op0, Op1, Op2, Op3><<<num_blocks, BLOCK_SIZE>>>(
        data0, data1, device_output, current_n, identity0, identity1, identity2);
    current_n = num_blocks;

    while (current_n > BLOCK_SIZE) {
        num_units = (current_n + STRIDE - 1) / STRIDE;
        num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

        quad_reduction_kernel_packed<T, Map1, Op0, Op1, Op2, Op3><<<num_blocks, BLOCK_SIZE>>>(
            device_output, device_output, current_n, identity0, identity1, identity2);

        current_n = num_blocks;
    }

    quad_reduction_kernel_packed<T, Map1, Op0, Op1, Op2, Op3><<<1, BLOCK_SIZE>>>(
        device_output, device_output, current_n, identity0, identity1, identity2);

    cudaMemcpy(out0, &device_output[0].red0, sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out1, &device_output[0].red1, sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out2, &device_output[0].red2, sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out3, &device_output[0].red3, sizeof(T), cudaMemcpyDeviceToDevice);

}