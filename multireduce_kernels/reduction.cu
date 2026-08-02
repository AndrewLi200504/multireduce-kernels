#include <stdio.h>
#include <cfloat>
#include <cstdint>

#include "single/single_reduction.h"
#include "double/double_reduction.h"
#include "triple/triple_reduction.h"
#include "quad/quad_reduction.h"
#include "triple/triple_reduction_dual_trimap.h"
#include "quad/quad_reduction_dual_quadmap.h"
void min_max_launcher(float* data, float* min, float* max, int n) {
    dual_reduction_launcher<float, dev_nop<float>, dev_nop<float>, dev_min<float>, dev_max<float>>(
        data, min, max, n, FLT_MAX, -FLT_MAX);
}


void sum_sumsq_launcher(float* data, float* sum, float* sumsq, int n) {
    dual_reduction_launcher<float, dev_nop<float>, dev_sqr<float>, dev_sum<float>, dev_sum<float>>(
        data, sum, sumsq, n, 0.0f, 0.0f);
}

void min_argmin_launcher(float* data, float* min, uint64_t* ind, int n) {
    single_reduction_launcher<float, uint64_t, dev_nop<float>, dev_min<float>>(
        data, min, ind, n, FLT_MAX, UINT64_MAX);
}

void max_argmax_launcher(float* data, float* max, uint64_t* ind, int n) {
    single_reduction_launcher<float, uint64_t, dev_nop<float>, dev_max<float>>(
        data, max, ind, n, -FLT_MAX, UINT64_MAX);
}

void a_ab_b_launcher(float* data0, float* data1, float* asum, float* absum, float* bsum, int n) {
    triple_reduction_launcher<float, dev_nop<float>, dev_mult<float>, dev_nop<float>,
    dev_sum<float>, dev_sum<float>, dev_sum<float>> (data0, data1, asum, absum, bsum, n, 0.0f, 0.0f, 0.0f);
}

void asq_ab_bsq_launcher(float* data0, float* data1, float* asumsq, float* absum, float* bsumsq, int n) {
    triple_reduction_launcher<float, dev_sqr<float>, dev_mult<float>, dev_sqr<float>, 
    dev_sum<float>, dev_sum<float>, dev_sum<float>> (data0, data1, asumsq, absum, bsumsq, n, 0.0f, 0.0f, 0.0f);
}

void a_sqrtab_b_launcher(float* data0, float* data1, float* asum, float* sqrtabsum, float* bsum, int n) {
    triple_reduction_launcher<float, dev_nop<float>, dev_mult_sqrt<float>, dev_nop<float>, 
    dev_sum<float>, dev_sum<float>, dev_sum<float>> (data0, data1, asum, sqrtabsum, bsum, n, 0.0f, 0.0f, 0.0f);
}


void tp_fp_fn_launcher(bool* data0, bool* data1, int* reduce_tp, int* reduce_fp, int* reduce_fn, int n) {
    triple_reduction_launcher<bool, int, dev_bitwise_and<bool>, dev_fp<bool>, dev_fn<bool>,
    dev_sum<int>, dev_sum<int>, dev_sum<int>>(data0, data1, reduce_tp, reduce_fp, reduce_fn, n, false, false, false);
}

void a_ab_b_asq_launcher(float* data0, float* data1, float* asum, float* absum, float* bsum, float* asumsq, int n) {
    quad_reduction_launcher<float, dev_nop<float>, dev_mult<float>, dev_nop<float>, dev_sqr<float>, 
    dev_sum<float>, dev_sum<float>, dev_sum<float>, dev_sum<float>> 
    (data0, data1, asum, absum, bsum, asumsq, n, 0.0f, 0.0f, 0.0f);
}

void tp_fp_fn_tn_launcher(bool* data0, bool* data1, int* reduce_tp, int* reduce_fp, int* reduce_fn, int* reduce_tn, int n) {
    quad_reduction_launcher<bool, int, dev_bitwise_and<bool>, dev_fp<bool>, dev_fn<bool>, dev_tn<bool>,
    dev_sum<int>, dev_sum<int>, dev_sum<int>, dev_sum<int>>(
        data0, data1, reduce_tp, reduce_fp, reduce_fn, reduce_tn, n, false, false, false, false
    );
}

