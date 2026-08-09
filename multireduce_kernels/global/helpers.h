#include <cmath>
template<typename T> __device__ __forceinline__ T dev_nop(T a) { return a; }
template<typename T> __device__ __forceinline__ T dev_sqr(T a) { return a * a; }
template<typename T> __device__ __forceinline__ T dev_min(T a, T b) { return a < b ? a : b; }
template<typename T> __device__ __forceinline__ T dev_max(T a, T b) { return a > b ? a : b; }
template<typename T> __device__ __forceinline__ T dev_sum(T a, T b) { return a + b; }
template<typename T> __device__ __forceinline__ T dev_mult(T a, T b) { return a * b; }
template<typename T> __device__ __forceinline__ T dev_mult_sqrt(T a, T b) { return sqrt(a) * sqrt(b); }
template<typename T> __device__ __forceinline__ T dev_bitwise_and(T a, T b) { return a & b; }
template<typename T> __device__ __forceinline__ T dev_fp(T a, T b) { return !a & b; }
template<typename T> __device__ __forceinline__ T dev_fn(T a, T b) { return a & !b; }
template<typename T> __device__ __forceinline__ T dev_tn(T a, T b) { return !a & !b; }
template<typename T> __device__ __forceinline__ T dev_log_weighted(T a) { return a * logf(a); }
template<typename T> __device__ __forceinline__ T dev_log_prod(T a, T b) { return a * logf(b); }
template<typename T, typename U> __device__ __forceinline__ U dev_isnan(T a) { return isnan(a); }
template<typename T, typename U> __device__ __forceinline__ U dev_isinf(T a) { return isinf(a); }
template<typename T, typename U> __device__ __forceinline__ U dev_iszero(T a) { return a == 0; }
template<typename T, typename U> __device__ __forceinline__ U dev_ispositive(T a) { return a > 0; }
template<typename T, typename U> __device__ __forceinline__ U dev_threshold(T a, T b) { return a > b; }
template<typename T, typename U> __device__ __forceinline__ U dev_range(T a, T b, T c) { return (b < a) && (a < c); }

