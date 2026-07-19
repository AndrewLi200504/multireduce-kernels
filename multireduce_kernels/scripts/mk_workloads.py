import multireduce_kernels as mk

def cosinesim(t0, t1):
    asq, ab, bsq = mk.asq_ab_bsq(t0, t1)
    return ab / (asq.sqrt() * bsq.sqrt())
def l2norm(t0, t1):
    asq, ab, bsq = mk.asq_ab_bsq(t0, t1)
    return (asq - 2 * ab + bsq).sqrt()
def covariance(t0, t1): 
    n = t0.numel()
    a, ab, b = mk.a_ab_b(t0, t1)
    return (ab - 1 / n * a * b) / (n - 1)
