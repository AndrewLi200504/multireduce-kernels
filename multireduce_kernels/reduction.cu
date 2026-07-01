#include <stdio.h>
#include <cfloat>

#include "reduction.h"



void min_max_launcher(float* data, float* min, float* max, int n) {
    dual_reduction_launcher<float, dev_nop<float>, dev_nop<float>, dev_min<float>, dev_max<float>>(
        data, min, max, n, FLT_MAX, -FLT_MAX);
}


void sum_sumsq_launcher(float* data, float* sum, float* sumsq, int n) {
    dual_reduction_launcher<float, dev_nop<float>, dev_sqr<float>, dev_sum<float>, dev_sum<float>>(
        data, sum, sumsq, n, 0.0f, 0.0f);
}