
template<typename T, typename U, U (*Map0)(T, T), U (*Map1)(T, T), U (*Map2)(T, T),
U (*Op0)(U, U), U (*Op1)(U, U), U (*Op2)(U, U)>
__global__ void triple_reduction_kernel(const T* a, const T b, const T c, const T d, triple_reduction<U>* blocktrs,
                                        int n, T identity0, T identity1, T identity2) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ triple_reduction<U> sdata[NUM_WARPS];

    triple_reduction<U> tr{identity0, identity1, identity2};
    for (int k = 0; k < STRIDE; k++) {
        int currIdx = STRIDE * i + k;
        if (currIdx < n) {
            tr.red0 = Op0(tr.red0, Map0(a[currIdx], b)); 
            tr.red1 = Op1(tr.red1, Map1(a[currIdx], c));
            tr.red2 = Op2(tr.red2, Map2(a[currIdx], d));
        } else {
            tr.red0 = Op0(tr.red0, identity0); 
            tr.red1 = Op1(tr.red1, identity1);
            tr.red2 = Op2(tr.red2, identity2);
        }
    }

    triple_reduction<U> warp_tr = warp_reduce_triple<U, Op0, Op1, Op2>(tr);
    if (lane == 0) sdata[warp_id] = warp_tr;
    __syncthreads();

    if (tid == 0) {
        triple_reduction<U> final_tr{identity0, identity1, identity2};
        for (int j = 0; j < NUM_WARPS; j++) {
            final_tr.red0 = Op0(final_tr.red0, sdata[j].red0);
            final_tr.red1 = Op1(final_tr.red1, sdata[j].red1);
            final_tr.red2 = Op2(final_tr.red2, sdata[j].red2);
        }
        blocktrs[blockIdx.x] = final_tr;
    }
}


template<typename T, typename U, U (*Map0)(T, T), U (*Map1)(T, T), U (*Map2)(T, T), U (*Op0)(U, U), U (*Op1)(U, U), U (*Op2)(U, U)>
void triple_reduction_launcher(const T* data, const T b, const T c, const T d, 
                            U* out0, U* out1, U* out2, int n, U identity0, 
                            U identity1, U identity2) {
    int current_n = n;
    int num_units = (current_n + STRIDE - 1) / STRIDE;
    int num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

    static thread_local triple_reduction<U>* device_output = nullptr;
    static thread_local size_t device_output_capacity = 0;

    if (num_blocks > device_output_capacity) {
        if (device_output) cudaFree(device_output);
        cudaMalloc(&device_output, num_blocks * sizeof(triple_reduction<U>));
        device_output_capacity = num_blocks;
    }

    triple_reduction_kernel<T, U, Map0, Map1, Map2, Op0, Op1, Op2><<<num_blocks, BLOCK_SIZE>>>(
        data, b, c, d, device_output, current_n, identity0, identity1, identity2);
    current_n = num_blocks;

    while (current_n > BLOCK_SIZE) {
        num_units = (current_n + STRIDE - 1) / STRIDE;
        num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

        triple_reduction_kernel_packed<U, Op0, Op1, Op2><<<num_blocks, BLOCK_SIZE>>>(
            device_output, device_output, current_n, identity0, identity1, identity2);

        current_n = num_blocks;
    }

    triple_reduction_kernel_packed<U, Op0, Op1, Op2><<<1, BLOCK_SIZE>>>(
        device_output, device_output, current_n, identity0, identity1, identity2);

    cudaMemcpy(out0, &device_output[0].red0, sizeof(U), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out1, &device_output[0].red1, sizeof(U), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out2, &device_output[0].red2, sizeof(U), cudaMemcpyDeviceToDevice);
}
