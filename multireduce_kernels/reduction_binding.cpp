#include <torch/extension.h>
#include <cstdint>
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

std::tuple<torch::Tensor, torch::Tensor> a_ab_binding(torch::Tensor input0, torch::Tensor input1) {
    TORCH_CHECK(input0.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input1.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input0.numel() == input1.numel(), "tensors must be the same length");

    auto asum = torch::empty({1}, input0.options());
    auto absum = torch::empty({1}, input0.options());
    a_ab_launcher(
        input0.data_ptr<float>(),
        input1.data_ptr<float>(),
        asum.data_ptr<float>(),
        absum.data_ptr<float>(),
        input0.numel()
    );
    return {asum, absum};
}


std::tuple<torch::Tensor, torch::Tensor> aloga_alogb_binding(torch::Tensor input0, torch::Tensor input1) {
    TORCH_CHECK(input0.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input1.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input0.numel() == input1.numel(), "tensors must be the same length");

    auto aloga_sum = torch::empty({1}, input0.options());
    auto alogb_sum = torch::empty({1}, input0.options());
    aloga_alogb_launcher(
        input0.data_ptr<float>(),
        input1.data_ptr<float>(),
        aloga_sum.data_ptr<float>(),
        alogb_sum.data_ptr<float>(),
        input0.numel()
    );
    return {aloga_sum, alogb_sum};
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tp_fp_fn_binding(torch::Tensor input0, torch::Tensor input1) {
    TORCH_CHECK(input0.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input1.is_cuda(), "must be cuda tensor");

    TORCH_CHECK(input0.numel() == input1.numel(), "tensors must be the same length");
    auto reduce_tp = torch::empty({1}, input0.options().dtype(torch::kInt));
    auto reduce_fp = torch::empty({1}, input0.options().dtype(torch::kInt));
    auto reduce_fn = torch::empty({1}, input0.options().dtype(torch::kInt));

    tp_fp_fn_launcher(
        input0.data_ptr<bool>(),
        input1.data_ptr<bool>(),
        reduce_tp.data_ptr<int>(),
        reduce_fp.data_ptr<int>(),
        reduce_fn.data_ptr<int>(),
        input0.numel()
    );
    return {reduce_tp, reduce_fp, reduce_fn};
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


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tp_fp_fn_tn_binding(torch::Tensor input0, torch::Tensor input1) {
    TORCH_CHECK(input0.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input1.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input0.numel() == input1.numel(), "tensors must be the same length");
    auto reduce_tp = torch::empty({1}, input0.options().dtype(torch::kInt));
    auto reduce_fp = torch::empty({1}, input0.options().dtype(torch::kInt));
    auto reduce_fn = torch::empty({1}, input0.options().dtype(torch::kInt));
    auto reduce_tn = torch::empty({1}, input0.options().dtype(torch::kInt));

    tp_fp_fn_tn_launcher(
        input0.data_ptr<bool>(),
        input1.data_ptr<bool>(),
        reduce_tp.data_ptr<int>(),
        reduce_fp.data_ptr<int>(),
        reduce_fn.data_ptr<int>(),
        reduce_tn.data_ptr<int>(),
        input0.numel()
    );
    return {reduce_tp, reduce_fp, reduce_fn, reduce_tn};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> a_ab_ac_ad_binding(torch::Tensor input0, torch::Tensor input1,
torch::Tensor input2, torch::Tensor input3) {
    TORCH_CHECK(input0.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input1.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input2.is_cuda(), "must be cuda tensor");
    TORCH_CHECK(input3.is_cuda(), "must be cuda tensor");

    TORCH_CHECK(input0.numel() == input1.numel(), "tensors must be the same length");
    TORCH_CHECK(input1.numel() == input2.numel(), "tensors must be the same length");
    TORCH_CHECK(input2.numel() == input3.numel(), "tensors must be the same length");

    auto asum = torch::empty({1}, input0.options());
    auto absum = torch::empty({1}, input0.options());
    auto acsum = torch::empty({1}, input0.options());
    auto adsum = torch::empty({1}, input0.options());

    a_ab_ac_ad_launcher(
        input0.data_ptr<float>(),
        input1.data_ptr<float>(),
        input2.data_ptr<float>(),
        input3.data_ptr<float>(),
        asum.data_ptr<float>(),
        absum.data_ptr<float>(),
        acsum.data_ptr<float>(),
        adsum.data_ptr<float>(),
        input0.numel()
    );
    return {asum, absum, acsum, adsum};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> nan_inf_zero_psive_binding(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "must be cuda tensor");

    auto nancnt = torch::empty({1}, input.options().dtype(torch::kInt));
    auto infcnt = torch::empty({1}, input.options().dtype(torch::kInt));
    auto zerocnt = torch::empty({1}, input.options().dtype(torch::kInt));
    auto psivecnt = torch::empty({1}, input.options().dtype(torch::kInt));

    nan_inf_zero_psive_launcher(
        input.data_ptr<float>(),
        nancnt.data_ptr<int>(),
        infcnt.data_ptr<int>(),
        zerocnt.data_ptr<int>(),
        psivecnt.data_ptr<int>(),
        input.numel()
    );
    return {nancnt, infcnt, zerocnt, psivecnt};
}



std::tuple<torch::Tensor, torch::Tensor> threshold_binding(torch::Tensor input, 
int threshold0, int threshold1) {
    TORCH_CHECK(input.is_cuda(), "must be cuda tensor");

    auto threshold0cnt = torch::empty({1}, input.options().dtype(torch::kInt));
    auto threshold1cnt = torch::empty({1}, input.options().dtype(torch::kInt));

    threshold_launcher(
        input.data_ptr<float>(),
        threshold0, 
        threshold1, 
        threshold0cnt.data_ptr<int>(),
        threshold1cnt.data_ptr<int>(),
        input.numel()
    );
    return {threshold0cnt, threshold1cnt};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> threshold_binding(torch::Tensor input, 
int threshold0, int threshold1, int threshold2) {
    TORCH_CHECK(input.is_cuda(), "must be cuda tensor");

    auto threshold0cnt = torch::empty({1}, input.options().dtype(torch::kInt));
    auto threshold1cnt = torch::empty({1}, input.options().dtype(torch::kInt));
    auto threshold2cnt = torch::empty({1}, input.options().dtype(torch::kInt));

    threshold_launcher(
        input.data_ptr<float>(),
        threshold0, 
        threshold1, 
        threshold2, 
        threshold0cnt.data_ptr<int>(),
        threshold1cnt.data_ptr<int>(),
        threshold2cnt.data_ptr<int>(),
        input.numel()
    );
    return {threshold0cnt, threshold1cnt, threshold2cnt};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> threshold_binding(torch::Tensor input, 
int threshold0, int threshold1, int threshold2, int threshold3) {
    TORCH_CHECK(input.is_cuda(), "must be cuda tensor");

    auto threshold0cnt = torch::empty({1}, input.options().dtype(torch::kInt));
    auto threshold1cnt = torch::empty({1}, input.options().dtype(torch::kInt));
    auto threshold2cnt = torch::empty({1}, input.options().dtype(torch::kInt));
    auto threshold3cnt = torch::empty({1}, input.options().dtype(torch::kInt));

    threshold_launcher(
        input.data_ptr<float>(),
        threshold0, 
        threshold1, 
        threshold2, 
        threshold3,
        threshold0cnt.data_ptr<int>(),
        threshold1cnt.data_ptr<int>(),
        threshold2cnt.data_ptr<int>(),
        threshold3cnt.data_ptr<int>(),
        input.numel()
    );
    return {threshold0cnt, threshold1cnt, threshold2cnt, threshold3cnt};
}

PYBIND11_MODULE(multireduce_kernels, m) {
    py::options options;
    options.disable_function_signatures();
    m.def("min_max", &min_max_binding, R"(
        Accepts:
            1. float32 tensor
        Returns:
            1. The min 
            2. The max
        metadata | {"type":"float", "args": 1, "return_tuple_size": 2, "reds": [{"min": -1}, {"max": -1}]}
        )");
    m.def("sum_sumsq", &sum_sumsq_binding, R"(
        Accepts:
            1. float32 tensor
        Returns:
            1. The sum of all elements
            2. The sum of squares of all elements
        metadata | {"type":"float", "args": 1, "return_tuple_size": 2, "reds": [{"sum": -1}, {"sumsq": -1}]}
        )");
    m.def("min_argmin", &min_argmin_binding, R"(
        Accepts: 
            1. float32 tensor
        Returns:
            1. The min
            2. The index at which the min occurs
        metadata | {"type":"float", "args": 1, "return_tuple_size": 2, "reds": [{"min": -1}, {"argmin": -1}]})");
    m.def("max_argmax", &max_argmax_binding, R"(
        Accepts:
            1. float32 tensor
        Returns:
            1. The max
            2. The index at which the max occurs
        metadata | {"type":"float", "args": 1,"return_tuple_size": 2, "reds": [{"max": -1}, {"argmax": -1}]})");
    m.def("a_ab", &a_ab_binding, R"(
        Accepts:
            1. float32 tensor
            2. float32 tensor
        Returns:
            1. The sum of the first tensor's elements 
            2. The sum of the products between corresponding elements from both tensors
        metadata | {"type":"float", "args": 2, "return_tuple_size": 2, "reds": [{"a": 0}, {"ab": -1}]})");
    m.def("aloga_alogb", &aloga_alogb_binding, R"(
        Accepts:
            1. float32 tensor
            2. float32 tensor
        Returns:
            1. The sum of each element multiplied by the ln of itself in the first tensor
            2. The sum of each element in the first tensor multiplied by the ln of the correspending element in the second
            tensor 
        metadata | {"type": "prob", "args": 2, "return_tuple_size": 2, "reds": [{"aloga": 0}, {"alogb": -1}]})");
    m.def("a_ab_b", &a_ab_b_binding, R"(
        Accepts:
            1. float32 tensor
            2. float32 tensor
        Returns:
            1. The sum of the first tensor's elements 
            2. The sum of the products between corresponding elements from both tensors
            3. The sum of the second tensor's elements
        metadata | {"type":"float", "args": 2, "return_tuple_size": 3, "reds": [{"a": 0}, {"ab": -1}, {"b": 1}]})");
    m.def("asq_ab_bsq", &asq_ab_bsq_binding, R"(
        Accepts:
            1. float32 tensor
            2. float32 tensor
        Returns:
            1. The sum of the squares of the first tensor's elements
            2. The sum of the products between corresponding  elements from both tensors
            3. The sum of the squares of the second tensor's elements
        metadata | {"type":"float", "args": 2, "return_tuple_size": 3, "reds": [{"asq": 0}, {"ab": -1}, {"bsq": 1}]})");
    m.def("a_sqrtab_b", &a_sqrtab_b_binding, R"(
        Accepts: 
            1. float32 tensor
            2. float32 tensor
        Returns:
            1. The sum of the squares of the first tensor's elements
            2. The sum of the products between corresponding elements' square roots from both tensors
            3. The sum of the squares of the second tensor's elements
        metadata | {"type":"float", "args": 2, "return_tuple_size": 3, "reds": [{"a": 0}, {"sqrtab": -1}, {"b": 1}]})");
    m.def("tp_fp_fn", &tp_fp_fn_binding, R"(
        Accepts:
            1. boolean tensor
            2. boolean tensor
        Returns:
            1. The number of corresponding elements from both tensor's that are both True
            2. The number of corresponding elements from both tensor's such that the element from the first tensor 
            is False and the element from the second tensor is True
            3. The number of corresponding elements from both tensor's such that the element from the first tensor
            is True and the element from the second tensor is False
        metadata | {"type":"bool", "args": 2, "return_tuple_size": 3, "reds": [{"tp": -1}, {"fp": -1}, {"fn": -1}]})");
    m.def("a_ab_b_asq", &a_ab_b_asq_binding, R"(
        Accepts:
            1. float32 tensor
            2. float32 tensor
        Returns:
            1. The sum of the first tensor's elements
            2. The sum of the products between corresponding elements from both tensors
            3. The sum of the second tensor's elements
            4. The sum of the squares of the first tensor's elements
        metadata | {"type":"float", "args": 2, "return_tuple_size": 4, "reds": [{"a": 0}, {"ab": -1}, {"b": 1}, {"asq": 0}]})");
    m.def("tp_fp_fn_tn", &tp_fp_fn_tn_binding, R"(
        Accepts:
            1. boolean tensor
            2. boolean tensor
        Returns:
            1. The number of corresponding elements from both tensor's that are both True
            2. The number of corresponding elements from both tensor's such that the element from the first tensor 
            is False and the element from the second tensor is True
            3. The number of corresponding elements from both tensor's such that the element from the first tensor
            is True and the element from the second tensor is False
            4. The number of corresponding elements from both tensor's that are both False
        metadata | {"type": "bool", "args": 2, "return_tuple_size": 4, "reds": [{"tp": -1}, {"fp": -1}, {"fn": -1}, {"tn": -1}]})");
    m.def("a_ab_ac_ad", &a_ab_ac_ad_binding, R"(
        Accepts:
            1. float32 tensor
            2. float32 tensor
            3. float32 tensor
            4. float32 tensor
        Returns: 
            1. The sum of the first tensor's elements 
            2. The sum of the products between corresponding elements from tensors 1 and 2
            3. The sum of the products between corresponding elements from tensors 1 and 3
            4. The sum of the products between corresponding elements from tensors 1 and 4
        metadata | {"type":"float", "args": 4, "return_tuple_size": 4, "reds": [{"a": 0}, {"ab": [0, 1]}, {"ac": [0, 2]}, {"ad": [0, 3]}]})");
    m.def("nan_inf_zero_psive", &nan_inf_zero_psive_binding, R"(
        Accepts:
            1. float32 tensor
        Returns:
            1. The number of NaN values in the tensor
            2. The number of positive/negative infinity values in the tensor
            3. The number of zeros in the tensor
            4. The number of positive values in the tensor
        metadata | {"type": "float_debug", "args": 1, "return_tuple_size": 4, "reds": [{"nan": -1}, {"inf": -1}, {"zero": -1}, {"psive": -1}]})");
    m.def("threshold", py::overload_cast<torch::Tensor, int, int, int, int>(&threshold_binding), 
        R"(
        Accepts: 
            1. float32 tensor 
            2. integer threshold
            3. integer threshold
            4. integer threshold
            5. integer threshold
        Returns: 
            1. The number of values above threshold 1 in the tensor
            2. The number of values above threshold 2 in the tensor
            3. The number of values above threshold 3 in the tensor
            4. The number of values above threshold 4 in the tensor

        metadata | {"type": "float", "args": 1, "return_tuple_size": 4, "reds": [{"thresh0": [0, 1]}, {"thresh1": [0, 2]}, {"thresh2": [0, 3]}, {"thresh3": [0, 4]}], "extra_args": [0, 1, 2, 3]})")
        .def("threshold", py::overload_cast<torch::Tensor, int, int, int>(&threshold_binding))
        .def("threshold", py::overload_cast<torch::Tensor, int, int>(&threshold_binding));

}   



