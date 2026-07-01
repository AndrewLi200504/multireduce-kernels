#include <torch/extension.h>

void min_max_launcher(float* data, float* min, float* max, int n);
void sum_sumsq_launcher(float* data, float* sum, float* sumsq, int n);
