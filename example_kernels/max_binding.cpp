#include <torch/extension.h>

void max_launcher(float* data, float* scalar, int n);

void min_max_launcher(float* data, float* min, float* max, int n);

torch::Tensor max_binding(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "must be cuda tensor");

    auto output = torch::empty({1}, input.options());

    max_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
    return output;
}


std::tuple<torch::Tensor, torch::Tensor> min_max_binding(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "must be cuda tensor");

    auto min = torch::empty({1}, input.options());
    auto max = torch::empty({1}, input.options());
    min_max_launcher(
        input.data_ptr<float>(),
        min.data_ptr<float>(),
        max.data_ptr<float>(),
        input.numel()
    );
    return {min, max};
}


PYBIND11_MODULE(example_kernels, m) {
    m.def("my_func", &max_binding, "does something");
    m.def("min_max", &min_max_binding, "does something");

}



