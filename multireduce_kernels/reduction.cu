#include <stdio.h>
#include <cfloat>
#include <cstdint>

#include "single_reduction.h"
#include "double_reduction.h"
#include "triple_reduction.h"


void min_max_launcher(float* data, float* min, float* max, int n) {
    dual_reduction_launcher<float, dev_nop<float>, dev_nop<float>, dev_min<float>, dev_max<float>>(
        data, min, max, n, FLT_MAX, -FLT_MAX);
}


void sum_sumsq_launcher(float* data, float* sum, float* sumsq, int n) {
    dual_reduction_launcher<float, dev_nop<float>, dev_sqr<float>, dev_sum<float>, dev_sum<float>>(
        data, sum, sumsq, n, 0.0f, 0.0f);
}

void min_argmin_launcher(float* data, float* min, uint64_t* ind, int n) {
    single_reduction_launcher<float, uint64_t, dev_nop<float>, dev_comp_min<float>,
    dev_comp_min<uint64_t>>(
        data, min, ind, n, FLT_MAX, UINT64_MAX);
}

void max_argmax_launcher(float* data, float* max, uint64_t* ind, int n) {
    single_reduction_launcher<float, uint64_t, dev_nop<float>, dev_comp_max<float>,
    dev_comp_min<uint64_t>>(
        data, max, ind, n, -FLT_MAX, UINT64_MAX);
}

void asq_ab_bsq_launcher(float* data0, float* data1, float* asumsq, float* absum, float* bsumsq, int n) {
    triple_reduction_launcher<float, dev_sqr<float>, dev_mult<float>, dev_sqr<float>, 
    dev_sum<float>, dev_sum<float>, dev_sum<float>> (data0, data1, asumsq, absum, bsumsq, n, 0.0f, 0.0f);
}