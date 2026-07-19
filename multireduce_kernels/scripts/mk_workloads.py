import multireduce_kernels as mk

def cosinesim(t0, t1):
    asq, ab, bsq = mk.asq_ab_bsq(t0, t1)
    return ab / (asq.sqrt() * bsq.sqrt())
def l2norm(t0, t1):
    asq, ab, bsq = mk.asq_ab_bsq(t0, t1)
    return (asq - 2 * ab + bsq).sqrt()
# def mk_cos_sim(asq, ab, bsq):
#     return ab / (asq.sqrt() * bsq.sqrt())
# def mk_l2_norm(asq, ab, bsq): 
#     return (asq - 2 * ab + bsq).sqrt()
# def mk_covariance(a, ab, b, n): 
#     return (ab - 1 / n * a * b) / (n - 1)