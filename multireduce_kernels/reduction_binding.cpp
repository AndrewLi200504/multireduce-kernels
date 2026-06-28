#include <torch/extension.h>
#include "reduction_binding.h"



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


PYBIND11_MODULE(multireduce_kernels, m) {
    m.def("min_max", &min_max_binding, "Return min and max");

}



