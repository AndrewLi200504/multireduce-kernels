#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define STRIDE 8

template<typename T, typename U>
struct single_reduction_ind {
    T red;
    U ind; 
};
template<typename T, typename U>
using sri = single_reduction_ind<T, U>;

template<typename T>
struct dual_reduction {
    T red0;
    T red1;
};

template<typename T> 
struct triple_reduction {
    T red0;
    T red1;
    T red2;
};

template<typename T> 
struct quad_reduction {
    T red0;
    T red1;
    T red2;
    T red3;
};


