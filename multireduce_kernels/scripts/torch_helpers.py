import torch
import torch.nn.functional as f
from constants import EPS 
def sum(t):
    return t.sum()
def sumsq(t):
    return (t ** 2).sum()
def aloga(t):
    return (t * t.log()).sum()
def summul(t0, t1):
    return (t0 * t1).sum()
def summulsqrt(t0, t1):
    return (t0.sqrt() * t1.sqrt()).sum()
def alogb(t0, t1):
    return (t0 * t1.log()).sum()
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

def precision(t0, t1):
    tp = intersection(t0, t1)
    _fp = fp(t0, t1)
    return tp / (tp + _fp)
def recall(t0, t1):
    tp = intersection(t0, t1)
    _fn = fn(t0, t1)
    return tp / (tp + _fn)

def kl_divergence(t0, t1):
    return f.kl_div(t1.log(), t0, reduction="sum")
def union(t0, t1):
    return (t0 | t1).sum()
def intersection(t0, t1):
    return (t0 & t1).sum()
def fp(t0, t1):
    return (~t0 & t1).sum()
def fn(t0, t1):
    return (t0 & ~t1).sum()
def tn(t0, t1):
    return (~t0 & ~t1).sum()
def iou(t0, t1):
    return intersection(t0, t1) / (union(t0, t1) + EPS)
def dice(t0, t1): 
    tp = intersection(t0, t1)
    return (2 * tp) / (2 * tp + fp(t0, t1) + fn(t0, t1) + EPS)


SPECIAL_REDS = {
    "asq": sumsq,
    "ab": summul,
    "ac": summul,
    "ad": summul,
    "bsq": sumsq,
    "sqrtab": summulsqrt,
    "a": sum,
    "b": sum,
    "tp": intersection,
    "acnt": sum,
    "bcnt": sum,
}