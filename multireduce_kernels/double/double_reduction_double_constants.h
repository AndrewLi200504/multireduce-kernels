
template<typename T, typename U, U (*Map0)(T, T, T), U (*Map1)(T, T, T),
U (*Op0)(U, U), U (*Op1)(U, U)>
__global__ void dual_reduction_kernel(const T* a, const T b, const T c, const T d, const T e, dual_reduction<U>* blockdrs,
                                        int n, T identity0, T identity1) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ dual_reduction<U> sdata[NUM_WARPS];

    dual_reduction<U> dr{identity0, identity1};
    for (int k = 0; k < STRIDE; k++) {
        int currIdx = STRIDE * i + k;
        if (currIdx < n) {
            dr.red0 = Op0(dr.red0, Map0(a[currIdx], b, c)); 
            dr.red1 = Op1(dr.red1, Map1(a[currIdx], d, e));
        } else {
            dr.red0 = Op0(dr.red0, identity0); 
            dr.red1 = Op1(dr.red1, identity1);
        }
    }

    dual_reduction<U> warp_dr = warp_reduce_dual<U, Op0, Op1>(dr);
    if (lane == 0) sdata[warp_id] = warp_dr;
    __syncthreads();

    if (tid == 0) {
        dual_reduction<U> final_dr{identity0, identity1};
        for (int j = 0; j < NUM_WARPS; j++) {
            final_dr.red0 = Op0(final_dr.red0, sdata[j].red0);
            final_dr.red1 = Op1(final_dr.red1, sdata[j].red1);
        }
        blockdrs[blockIdx.x] = final_dr;
    }
}


template<typename T, typename U, U (*Map0)(T, T, T), U (*Map1)(T, T, T), U (*Op0)(U, U), U (*Op1)(U, U)>
void dual_reduction_launcher(const T* data, const T b, const T c, const T d, const T e, U* out0, U* out1, int n, U identity0, 
                            U identity1) {
    int current_n = n;
    int num_units = (current_n + STRIDE - 1) / STRIDE;
    int num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

    static thread_local dual_reduction<U>* device_output = nullptr;
    static thread_local size_t device_output_capacity = 0;

    if (num_blocks > device_output_capacity) {
        if (device_output) cudaFree(device_output);
        cudaMalloc(&device_output, num_blocks * sizeof(dual_reduction<U>));
        device_output_capacity = num_blocks;
    }

    dual_reduction_kernel<T, U, Map0, Map1, Op0, Op1><<<num_blocks, BLOCK_SIZE>>>(
        data, b, c, d, e, device_output, current_n, identity0, identity1);
    current_n = num_blocks;

    while (current_n > BLOCK_SIZE) {
        num_units = (current_n + STRIDE - 1) / STRIDE;
        num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

        dual_reduction_kernel_packed<U, Op0, Op1><<<num_blocks, BLOCK_SIZE>>>(
            device_output, device_output, current_n, identity0, identity1);

        current_n = num_blocks;
    }

    dual_reduction_kernel_packed<U, Op0, Op1><<<1, BLOCK_SIZE>>>(
        device_output, device_output, current_n, identity0, identity1);

    cudaMemcpy(out0, &device_output[0].red0, sizeof(U), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out1, &device_output[0].red1, sizeof(U), cudaMemcpyDeviceToDevice);
}
