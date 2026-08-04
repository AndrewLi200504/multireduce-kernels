import multireduce_kernels as mk
from constants import EPS
def add_init_type(init_type_name: str):
    def wrapper(func):
        func.init_type_name = init_type_name
        return func
    return wrapper

@add_init_type("float")
def samplevar(t): 
    sum, sumsq = mk.sum_sumsq(t)
    return (sumsq - sum ** 2 / t.numel()) / (t.numel() - 1)
@add_init_type("float")
def cosinesim(t0, t1):
    asq, ab, bsq = mk.asq_ab_bsq(t0, t1)
    return ab / (asq.sqrt() * bsq.sqrt())
@add_init_type("float")
def l2norm(t0, t1):
    asq, ab, bsq = mk.asq_ab_bsq(t0, t1)
    return (asq - 2 * ab + bsq).sqrt()
@add_init_type("float")
def covariance(t0, t1): 
    n = t0.numel()
    a, ab, b = mk.a_ab_b(t0, t1)
    return (ab - 1 / n * a * b) / (n - 1)
@add_init_type("float")
def olsslope(t0, t1):
    a, ab, b, asq = mk.a_ab_b_asq(t0, t1)
    return (t0.numel() * ab - a * b) / (t0.numel() * asq - a ** 2)

@add_init_type("bool")
def iou(t0, t1):
    tp, fp, fn = mk.tp_fp_fn(t0, t1)
    return tp / (tp + fp + fn + EPS)
@add_init_type("bool") 
def dice(t0, t1):
    tp, fp, fn = mk.tp_fp_fn(t0, t1)
    return 2 * tp / (2 * tp + fp + fn + EPS)

@add_init_type("bool")
def precision(t0, t1):
    tp, fp, _ = mk.tp_fp_fn(t0, t1)
    return tp / (tp + fp + EPS)

@add_init_type("bool")
def recall(t0, t1):
    tp, _, fn = mk.tp_fp_fn(t0, t1)
    return tp / (tp + fn + EPS)

@add_init_type("prob")
def kl_divergence(t0, t1):
    plogp_sum, plogq_sum = mk.aloga_alogb(t0, t1)
    return plogp_sum - plogq_sum
