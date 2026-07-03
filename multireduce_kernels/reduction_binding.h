#include <torch/extension.h>
#include <cstdint>

void min_max_launcher(float* data, float* min, float* max, int n);
void sum_sumsq_launcher(float* data, float* sum, float* sumsq, int n);
void min_argmin_launcher(float* data, float* min, uint64_t* ind, int n);
void max_argmax_launcher(float* data, float* max, uint64_t* ind, int n);
