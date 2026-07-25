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

std::tuple<torch::Tensor, torch::Tensor> union_intersection_binding(torch::Tensor input0, torch::Tensor input1) {
    TORCH_CHECK(input0.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input1.is_cuda(), "must be cuda tensor");

    auto reduce_or = torch::empty({1}, input0.options().dtype(torch::kInt));
    auto reduce_and = torch::empty({1}, input0.options().dtype(torch::kInt));
    union_intersection_launcher(
        input0.data_ptr<bool>(),
        input1.data_ptr<bool>(),
        reduce_or.data_ptr<int>(),
        reduce_and.data_ptr<int>(),
        input0.numel()
    );
    return {reduce_or, reduce_and};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> a_ab_b_binding(torch::Tensor input0, torch::Tensor input1) {
    TORCH_CHECK(input0.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input1.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input0.numel() == input1.numel(), "tensors must be the same length");
    auto asum = torch::empty({1}, input0.options());
    auto absum = torch::empty({1}, input0.options());
    auto bsum = torch::empty({1}, input0.options());
    a_ab_b_launcher(
        input0.data_ptr<float>(),
        input1.data_ptr<float>(),
        asum.data_ptr<float>(),
        absum.data_ptr<float>(),
        bsum.data_ptr<float>(),
        input0.numel()
    );
    return {asum, absum, bsum};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> asq_ab_bsq_binding(torch::Tensor input0, torch::Tensor input1) {
    TORCH_CHECK(input0.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input1.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input0.numel() == input1.numel(), "tensors must be the same length");
    auto asumsq = torch::empty({1}, input0.options());
    auto absum = torch::empty({1}, input0.options());
    auto bsumsq = torch::empty({1}, input0.options());
    asq_ab_bsq_launcher(
        input0.data_ptr<float>(),
        input1.data_ptr<float>(),
        asumsq.data_ptr<float>(),
        absum.data_ptr<float>(),
        bsumsq.data_ptr<float>(),
        input0.numel()
    );
    return {asumsq, absum, bsumsq};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> a_sqrtab_b_binding(torch::Tensor input0, torch::Tensor input1) {
    TORCH_CHECK(input0.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input1.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input0.numel() == input1.numel(), "tensors must be the same length");
    auto asum = torch::empty({1}, input0.options());
    auto sqrtabsum = torch::empty({1}, input0.options());
    auto bsum = torch::empty({1}, input0.options());
    a_sqrtab_b_launcher(
        input0.data_ptr<float>(),
        input1.data_ptr<float>(),
        asum.data_ptr<float>(),
        sqrtabsum.data_ptr<float>(),
        bsum.data_ptr<float>(),
        input0.numel()
    );
    return {asum, sqrtabsum, bsum};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> a_ab_b_asq_binding(torch::Tensor input0, torch::Tensor input1) {
    TORCH_CHECK(input0.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input1.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input0.numel() == input1.numel(), "tensors must be the same length");
    auto asum = torch::empty({1}, input0.options());
    auto absum = torch::empty({1}, input0.options());
    auto bsum = torch::empty({1}, input0.options());
    auto asumsq = torch::empty({1}, input0.options());

    a_ab_b_asq_launcher(
        input0.data_ptr<float>(),
        input1.data_ptr<float>(),
        asum.data_ptr<float>(),
        absum.data_ptr<float>(),
        bsum.data_ptr<float>(),
        asumsq.data_ptr<float>(),
        input0.numel()
    );
    return {asum, absum, bsum, asumsq};
}


PYBIND11_MODULE(multireduce_kernels, m) {
    py::options options;
    options.disable_function_signatures();

    m.def("min_max", &min_max_binding, R"({"type":"float", "args": 1, "return_tuple_size": 2, "reds": [{"min": -1}, {"max": -1}]})");
    m.def("sum_sumsq", &sum_sumsq_binding, R"({"type":"float", "args": 1, "return_tuple_size": 2, "reds": [{"sum": -1}, {"sumsq": -1}]})");
    m.def("min_argmin", &min_argmin_binding, R"({"type":"float", "args": 1, "return_tuple_size": 2, "reds": [{"min": -1}, {"argmin": -1}]})");
    m.def("max_argmax", &max_argmax_binding, R"({"type":"float", "args": 1,"return_tuple_size": 2, "reds": [{"max": -1}, {"argmax": -1}]})");
    m.def("union_intersection", &union_intersection_binding, R"({"type":"bool", "args": 2, "return_tuple_size": 2, "reds": [{"union": -1}, {"intersection": -1}]})");
    m.def("a_ab_b", &a_ab_b_binding, R"({"type":"float", "args": 2, "return_tuple_size": 3, "reds": [{"a": 0}, {"ab": -1}, {"b": 1}]})");
    m.def("asq_ab_bsq", &asq_ab_bsq_binding, R"({"type":"float", "args": 2, "return_tuple_size": 3, "reds": [{"asq": 0}, {"ab": -1}, {"bsq": 1}]})");
    m.def("a_sqrtab_b", &a_sqrtab_b_binding, R"({"type":"float", "args": 2, "return_tuple_size": 3, "reds": [{"a": 0}, {"sqrtab": -1}, {"b": 1}]})");
    m.def("a_ab_b_asq", &a_ab_b_asq_binding, R"({"type":"float", "args": 2, "return_tuple_size": 4, "reds": [{"a": 0}, {"ab": -1}, {"b": 1}, {"asq": 0}]})");

}



