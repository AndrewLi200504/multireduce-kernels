import torch

def sumsq(t):
    return (t ** 2).sum()
def summul(t0, t1):
    return (t0 * t1).sum()
def samplevar(t): 
    return t.var()
def cosinesim(t0, t1): 
    return torch.dot(t0, t1) / (torch.linalg.vector_norm(t0) * torch.linalg.vector_norm(t1))
def l2norm(t0, t1):
    return torch.linalg.vector_norm(t0 - t1)
def covariance(t0, t1): 
    t0_centered = t0 - t0.mean()
    t1_centered = t1 - t1.mean()
    n = t0.numel()
    return torch.dot(t0_centered, t1_centered) / (n - 1)
SPECIAL_REDS = {
    "sumsq": sumsq,
    "asq": sumsq,
    "ab": summul,
    "bsq": sumsq,
    "samplevar": samplevar,
    "cosinesim": cosinesim,
    "l2norm": l2norm,
    "covariance": covariance
}