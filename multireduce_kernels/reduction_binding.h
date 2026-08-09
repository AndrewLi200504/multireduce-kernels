#include <torch/extension.h>
#include <cstdint>

void min_max_launcher(float* data, float* min, float* max, int n);
void sum_sumsq_launcher(float* data, float* sum, float* sumsq, int n);
void min_argmin_launcher(float* data, float* min, uint64_t* ind, int n);
void max_argmax_launcher(float* data, float* max, uint64_t* ind, int n);
void a_ab_launcher(float* data0, float* data1, float* asum, float* absum, int n);
void aloga_alogb_launcher(float* data0, float* data1, float* aloga_sum, float* alogb_sum, int n);
void a_ab_b_launcher(float* data0, float* data1, float* asum, float* absum, float* bsum, int n);
void asq_ab_bsq_launcher(float* data0, float* data1, float* asumsq, float* absum, float* bsumsq, int n);
void a_sqrtab_b_launcher(float* data0, float* data1, float* asum, float* sqrtabsum, float* bsum, int n);
void tp_fp_fn_launcher(bool* data0, bool* data1, int* reduce_tp, int* reduce_fp, int* reduce_fn, int n);
void a_ab_b_asq_launcher(float* data0, float* data1, float* asum, float* absum, float* bsum, float* asumsq, int n);
void tp_fp_fn_tn_launcher(bool* data0, bool* data1, int* reduce_tp, int* reduce_fp, int* reduce_fn, int* reduce_tn, int n);
void a_ab_ac_ad_launcher(float* data0, float* data1, float* data2, float* data3, float* asum, float* absum, 
float* acsum, float* adsum, int n);
void nan_inf_zero_psive_launcher(float* data, int* nancnt, int* infcnt, int* zerocnt, int* psivecnt, int n);
void threshold_launcher(float* data, int threshold0, int threshold1, int* threshold0cnt,
int* threshold1cnt, int n);
void threshold_launcher(float* data, int threshold0, int threshold1, int threshold2, int* threshold0cnt,
int* threshold1cnt, int* threshold2cnt, int n);
void threshold_launcher(float* data, int threshold0, int threshold1, int threshold2, int threshold3, 
int* threshold0cnt, int* threshold1cnt, int* threshold2cnt, int* threshold3cnt, int n);
void range_launcher(float* data, int lowerbound0, int upperbound0, int lowerbound1, int upperbound1,
int* range0cnt, int* range1cnt, int n);
void range_launcher(float* data, int lowerbound0, int upperbound0, int lowerbound1, int upperbound1,
int lowerbound2, int upperbound2, int* range0cnt, int* range1cnt, int* range2cnt, int n);
void range_launcher(float* data, int lowerbound0, int upperbound0, int lowerbound1, int upperbound1,
int lowerbound2, int upperbound2, int lowerbound3, int upperbound3, int* range0cnt, int* range1cnt, 
int* range2cnt, int* range3cnt, int n);