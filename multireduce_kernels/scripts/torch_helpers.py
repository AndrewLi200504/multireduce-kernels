import torch

def sum(t):
    return t.sum()
def sumsq(t):
    return (t ** 2).sum()
def summul(t0, t1):
    return (t0 * t1).sum()
def summulsqrt(t0, t1):
    return (t0.sqrt() * t1.sqrt()).sum()
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
def olsslope(t0, t1):
    t0_c = t0 - t0.mean()
    t1_c = t1 - t1.mean()

    return torch.dot(t0_c, t1_c) / torch.dot(t0_c, t0_c)

def union(t0, t1):
    return (t0 | t1).sum()
def intersection(t0, t1):
    return (t0 & t1).sum()
def fp(t0, t1):
    return (~t0 & t1).sum()
def fn(t0, t1):
    return (t0 & ~t1).sum()
def iou(t0, t1):
    return (intersection(t0, t1) + 1e-6) / (union(t0, t1) + 1e-6)
SPECIAL_REDS = {
    "sumsq": sumsq,
    "asq": sumsq,
    "ab": summul,
    "bsq": sumsq,
    "sqrtab": summulsqrt,
    "samplevar": samplevar,
    "cosinesim": cosinesim,
    "l2norm": l2norm,
    "covariance": covariance,
    "olsslope": olsslope,
    "a": sum,
    "b": sum,
    "union": union,
    "intersection": intersection,
    "tp": intersection,
    "fp": fp,
    "fn": fn,
    "iou": iou
}