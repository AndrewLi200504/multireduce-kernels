#include <torch/extension.h>
#include "reduction_binding.h"
#include <cstdint>



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

std::tuple<torch::Tensor, torch::Tensor> sum_sumsq_binding(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "must be cuda tensor");

    auto sum = torch::empty({1}, input.options());
    auto sumsq = torch::empty({1}, input.options());
    sum_sumsq_launcher(
        input.data_ptr<float>(),
        sum.data_ptr<float>(),
        sumsq.data_ptr<float>(),
        input.numel()
    );
    return {sum, sumsq};
}

std::tuple<torch::Tensor, torch::Tensor> min_argmin_binding(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "must be cuda tensor");

    auto min = torch::empty({1}, input.options());
    auto argmin = torch::empty(
        {1},
        input.options().dtype(torch::kUInt64)
    );
    min_argmin_launcher(
        input.data_ptr<float>(),
        min.data_ptr<float>(),
        argmin.data_ptr<uint64_t>(),
        input.numel()
    );
    return {min, argmin};
}
std::tuple<torch::Tensor, torch::Tensor> max_argmax_binding(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "must be cuda tensor");

    auto max = torch::empty({1}, input.options());
    auto argmax = torch::empty(
        {1},
        input.options().dtype(torch::kUInt64)
    );
    max_argmax_launcher(
        input.data_ptr<float>(),
        max.data_ptr<float>(),
        argmax.data_ptr<uint64_t>(),
        input.numel()
    );
    return {max, argmax};
}

PYBIND11_MODULE(multireduce_kernels, m) {
    m.def("min_max", &min_max_binding, "Return min and max");
    m.def("sum_sumsq", &sum_sumsq_binding, "Return sum and sum of squares");
    m.def("min_argmin", &min_argmin_binding, "Return min and argmin");
    m.def("max_argmax", &max_argmax_binding, "Return max and argmax");
}



