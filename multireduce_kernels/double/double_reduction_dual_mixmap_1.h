
template<typename T, T (*Map0)(T), T (*Map1)(T, T), T (*Op0)(T, T), T (*Op1)(T, T)>
__global__ void dual_reduction_kernel(const T* a, const T* b, dual_reduction<T>* blockdrs,
                                        int n, T identity0, T identity1) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ dual_reduction<T> sdata[NUM_WARPS];

    dual_reduction<T> dr{identity0, identity1};
    for (int k = 0; k < STRIDE; k++) {
        int currIdx = STRIDE * i + k;
        if (currIdx < n) {
            dr.red0 = Op0(dr.red0, Map0(a[currIdx]));
            dr.red1 = Op1(dr.red1, Map1(a[currIdx], b[currIdx]));
        } else {
            dr.red0 = Op0(dr.red0, identity0);
            dr.red1 = Op1(dr.red1, identity1);
        }
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

template<typename T, T (*Map0)(T), T (*Map1)(T, T), T (*Op0)(T, T), T (*Op1)(T, T)>
void dual_reduction_launcher(const T* data0, const T* data1, T* out0, T* out1, int n,
                              T identity0, T identity1) {
    int current_n = n;
    int num_units = (current_n + STRIDE - 1) / STRIDE;
    int num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

    static thread_local dual_reduction<T>* device_output = nullptr;
    static thread_local size_t device_output_capacity = 0;

    if (num_blocks > device_output_capacity) {
        if (device_output) cudaFree(device_output);
        cudaMalloc(&device_output, num_blocks * sizeof(dual_reduction<T>));
        device_output_capacity = num_blocks;
    }

    dual_reduction_kernel<T, Map0, Map1, Op0, Op1><<<num_blocks, BLOCK_SIZE>>>(
        data0, data1, device_output, current_n, identity0, identity1);
    current_n = num_blocks;

    while (current_n > BLOCK_SIZE) {
        num_units = (current_n + STRIDE - 1) / STRIDE;
        num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

        dual_reduction_kernel_packed<T, Op0, Op1><<<num_blocks, BLOCK_SIZE>>>(
            device_output, device_output, current_n, identity0, identity1);

        current_n = num_blocks;
    }

    dual_reduction_kernel_packed<T, Op0, Op1><<<1, BLOCK_SIZE>>>(
        device_output, device_output, current_n, identity0, identity1);

    cudaMemcpy(out0, &device_output[0].red0, sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out1, &device_output[0].red1, sizeof(T), cudaMemcpyDeviceToDevice);
}