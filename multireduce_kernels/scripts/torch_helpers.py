import torch
def sumsq(t):
    return (t ** 2).sum()
def summul(t0, t1):
    return (t0 * t1).sum()
def cosinesim(t0, t1): 
    #return F.cosine_similarity(t0, t1, dim=0)
    #Fastest version of cosine_similarity on pytorch
    return torch.dot(t0, t1) / (torch.linalg.vector_norm(t0) * torch.linalg.vector_norm(t1))

def l2norm(t0, t1):
    return torch.linalg.vector_norm(t0 - t1)
SPECIAL_REDS = {
    "sumsq": sumsq,
    "asq": sumsq,
    "ab": summul,
    "bsq": sumsq,
    "cosinesim": cosinesim,
    "l2norm": l2norm
}