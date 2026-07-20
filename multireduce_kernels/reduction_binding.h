#include <torch/extension.h>
#include <cstdint>

void min_max_launcher(float* data, float* min, float* max, int n);
void sum_sumsq_launcher(float* data, float* sum, float* sumsq, int n);
void min_argmin_launcher(float* data, float* min, uint64_t* ind, int n);
void max_argmax_launcher(float* data, float* max, uint64_t* ind, int n);
void a_ab_b_launcher(float* data0, float* data1, float* asum, float* absum, float* bsum, int n);
void asq_ab_bsq_launcher(float* data0, float* data1, float* asumsq, float* absum, float* bsumsq, int n);
void a_sqrtab_b_launcher(float* data0, float* data1, float* asum, float* sqrtabsum, float* bsum, int n);
void a_ab_b_asq_launcher(float* data0, float* data1, float* asum, float* absum, float* bsum, float* asumsq, int n);