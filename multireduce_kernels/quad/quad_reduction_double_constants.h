
template<typename T, typename U, U (*Map0)(T, T, T), U (*Map1)(T, T, T), U (*Map2)(T, T, T), U (*Map3)(T, T, T),
U (*Op0)(U, U), U (*Op1)(U, U), U (*Op2)(U, U), U (*Op3)(U, U)>
__global__ void quad_reduction_kernel(const T* a, const T b, const T c, const T d, const T e, const T f, const T g, 
    const T h, const T l, quad_reduction<U>* blockqrs, int n, T identity0, T identity1, T identity2, T identity3) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ quad_reduction<U> sdata[NUM_WARPS];

    quad_reduction<U> qr{identity0, identity1, identity2, identity3};
    for (int k = 0; k < STRIDE; k++) {
        int currIdx = STRIDE * i + k;
        if (currIdx < n) {
            qr.red0 = Op0(qr.red0, Map0(a[currIdx], b, c)); 
            qr.red1 = Op1(qr.red1, Map1(a[currIdx], d, e));
            qr.red2 = Op2(qr.red2, Map2(a[currIdx], f, g));
            qr.red3 = Op3(qr.red3, Map3(a[currIdx], h, l));

        } else {
            qr.red0 = Op0(qr.red0, identity0); 
            qr.red1 = Op1(qr.red1, identity1);
            qr.red2 = Op2(qr.red2, identity2);
            qr.red3 = Op3(qr.red3, identity3);
        }
    }

    quad_reduction<U> warp_qr = warp_reduce_quad<U, Op0, Op1, Op2, Op3>(qr);
    if (lane == 0) sdata[warp_id] = warp_qr;
    __syncthreads();

    if (tid == 0) {
        quad_reduction<U> final_qr{identity0, identity1, identity2, identity3};
        for (int j = 0; j < NUM_WARPS; j++) {
            final_qr.red0 = Op0(final_qr.red0, sdata[j].red0);
            final_qr.red1 = Op1(final_qr.red1, sdata[j].red1);
            final_qr.red2 = Op2(final_qr.red2, sdata[j].red2);
            final_qr.red3 = Op3(final_qr.red3, sdata[j].red3);
        }
        blockqrs[blockIdx.x] = final_qr;
    }
}


template<typename T, typename U, U (*Map0)(T, T, T), U (*Map1)(T, T, T), U (*Map2)(T, T, T), U (*Map3)(T, T, T),
U (*Op0)(U, U), U (*Op1)(U, U), U (*Op2)(U, U), U (*Op3)(U, U)>
void quad_reduction_launcher(const T* data, const T b, const T c, const T d, const T e, const T f, const T g, const T h,
                            const T i, U* out0, U* out1, U* out2, U *out3, int n, U identity0, U identity1, U identity2,
                            U identity3) {
    int current_n = n;
    int num_units = (current_n + STRIDE - 1) / STRIDE;
    int num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

    static thread_local quad_reduction<U>* device_output = nullptr;
    static thread_local size_t device_output_capacity = 0;

    if (num_blocks > device_output_capacity) {
        if (device_output) cudaFree(device_output);
        cudaMalloc(&device_output, num_blocks * sizeof(quad_reduction<U>));
        device_output_capacity = num_blocks;
    }

    quad_reduction_kernel<T, U, Map0, Map1, Map2, Map3, Op0, Op1, Op2, Op3><<<num_blocks, BLOCK_SIZE>>>(
        data, b, c, d, e, f, g, h, i, device_output, current_n, identity0, identity1, identity2, identity3);
    current_n = num_blocks;

    while (current_n > BLOCK_SIZE) {
        num_units = (current_n + STRIDE - 1) / STRIDE;
        num_blocks = (num_units + BLOCK_SIZE - 1) / BLOCK_SIZE;

        quad_reduction_kernel_packed<U, Op0, Op1, Op2, Op3><<<num_blocks, BLOCK_SIZE>>>(
            device_output, device_output, current_n, identity0, identity1, identity2, identity3);

        current_n = num_blocks;
    }

    quad_reduction_kernel_packed<U, Op0, Op1, Op2, Op3><<<1, BLOCK_SIZE>>>(
        device_output, device_output, current_n, identity0, identity1, identity2, identity3);

    cudaMemcpy(out0, &device_output[0].red0, sizeof(U), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out1, &device_output[0].red1, sizeof(U), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out2, &device_output[0].red2, sizeof(U), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out3, &device_output[0].red3, sizeof(U), cudaMemcpyDeviceToDevice);

}
