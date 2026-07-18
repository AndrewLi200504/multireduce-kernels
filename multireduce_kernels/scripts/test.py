from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import multireduce_kernels as mk

from collections import namedtuple
from typing import Any, Callable, NamedTuple, Optional
from contextlib import contextmanager
from time import perf_counter

@contextmanager
def timeit():
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"Elapsed time: {end - start:.2f} seconds")
def sumsq(t):
    return (t ** 2).sum()


def cos_sim(t0, t1): 
    #return F.cosine_similarity(t0, t1, dim=0)
    #Fastest version of cosine_similarity on pytorch
    return torch.dot(t0, t1) / (torch.linalg.vector_norm(t0) * torch.linalg.vector_norm(t1))
def l2_norm(t0, t1):
    return torch.linalg.vector_norm(t0 - t1)
def covariance(t0, t1): 
    #return torch.cov((torch.stack((t0, t1)))[1][0]
    t0_centered = t0 - t0.mean()
    t1_centered = t1 - t1.mean()
    
    # Calculate covariance (N-1 for sample, N for population)
    n = t0.numel()
    return torch.dot(t0_centered, t1_centered) / (n - 1)
def mk_cos_sim(asq, ab, bsq):
    return ab / (asq.sqrt() * bsq.sqrt())
def mk_l2_norm(asq, ab, bsq): 
    return torch.sqrt(asq - 2 * ab + bsq)
def mk_covariance(a, ab, b, n): 
    return (ab - 1 / n * a * b) / (n - 1)

class MkDoubleReduction(NamedTuple): 
    red0: Any
    red1: Any
class DoubleReduction(NamedTuple): 
    op0: Callable[[Any], Any]
    op1: Callable[[Any], Any]
    fusedred: Callable[[Any], MkDoubleReduction]
double_red_dict: dict[str, DoubleReduction] = {
    "min_max": DoubleReduction(op0=torch.min, op1=torch.max, fusedred=mk.min_max),
    "min_argmin": DoubleReduction(op0=torch.min, op1=torch.argmin, fusedred=mk.min_argmin),
    "max_argmax": DoubleReduction(op0=torch.max, op1=torch.argmax, fusedred=mk.max_argmax),
    "sum_sumsq": DoubleReduction(op0=torch.sum, op1=sumsq, fusedred=mk.sum_sumsq)
}
class MkTripleReduction(NamedTuple): 
    red0: Any
    red1: Any
    red2: Any
class TripleReduction(NamedTuple): 
    original: Callable[[Any, Any], Any]
    fusedred: Callable[[Any, Any], MkTripleReduction]
    mk_reducer: Callable[..., Any]
#TripleReduction = namedtuple('TripleReduction', ['original', 'fusedred', 'mk_reducer'])
triple_red_dict: dict[str, TripleReduction] = {
    "cosine_similarity": TripleReduction(original=cos_sim, fusedred=mk.asq_ab_bsq, mk_reducer=mk_cos_sim),
    "covariance": TripleReduction(original=covariance, fusedred=mk.a_ab_b, mk_reducer=mk_covariance),
    "l2_norm": TripleReduction(original=l2_norm, fusedred=mk.asq_ab_bsq, mk_reducer=mk_l2_norm)
}
def double_red(red_key):

    torch_total_time = 0
    mk_total_time = 0
    inacc_op0 = 0
    inacc_op1 = 0
    red = double_red_dict[red_key]
    for _ in range(100):
        t = torch.randn(10000000).to("cuda")
        torch.cuda.synchronize()
        start_torch = perf_counter()
        op0_torch = red.op0(t)
        op1_torch = red.op1(t)
        torch.cuda.synchronize()
        end_torch = perf_counter()
        torch_total_time += (end_torch - start_torch)
        torch.cuda.synchronize()
        start_mk = perf_counter()
        op0_mk, op1_mk = red.fusedred(t)
        torch.cuda.synchronize()
        end_mk = perf_counter()
        mk_total_time += (end_mk - start_mk)
        if abs(op0_mk - op0_torch) < 1e-2:
            inacc_op0 += 1
        op1_mk = op1_mk.to(torch.long) if op1_mk.dtype is torch.uint64 else op1_mk

        if abs(op1_mk - op1_torch) < 1e-2:
            inacc_op1 += 1
    torch_avg_time = torch_total_time / 100
    mk_avg_time = mk_total_time / 100
    op0_accuracy = inacc_op0 / 100
    op1_accuracy = inacc_op1 / 100
    print(f"Average time on torch: {torch_avg_time}")
    print(f"Average time on mk: {mk_avg_time}")
    print(f"{red.op0.__name__} accuracy: {op0_accuracy}")
    print(f"{red.op1.__name__} accuracy: {op1_accuracy}")
def triple_red(red_key): 
    extra_args = ()
    if red_key in ["covariance"]:
        extra_args = (10000000,)
    torch_total_time = 0
    mk_total_time = 0
    inacc_red = 0
    red = triple_red_dict[red_key]
    for _ in range(100):
        t0 = torch.randn(10000000).to("cuda")
        t1 = torch.randn(10000000).to("cuda")
        torch.cuda.synchronize()
        start_torch = perf_counter()
        triple_red_torch = red.original(t0, t1)
        torch.cuda.synchronize()
        end_torch = perf_counter()
        torch_total_time += (end_torch - start_torch)
        torch.cuda.synchronize()
        start_mk = perf_counter()
        triple_red_mk = red.mk_reducer(*red.fusedred(t0, t1), *extra_args)
        torch.cuda.synchronize()
        end_mk = perf_counter()
        mk_total_time += (end_mk - start_mk)
        if abs(triple_red_torch - triple_red_mk) < 1e-2:
            inacc_red += 1
    torch_avg_time = torch_total_time / 100
    mk_avg_time = mk_total_time / 100
    red_accuracy = inacc_red / 100
    print(f"Average time on torch: {torch_avg_time}")
    print(f"Average time on mk: {mk_avg_time}")
    print(f"{red.original.__name__} accuracy: {red_accuracy}")
    pass
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("red_key", help="Fused red_dict to use") 
    args = parser.parse_args()
    if args.red_key in double_red_dict: 
        double_red(args.red_key)
    elif args.red_key in triple_red_dict:
        triple_red(args.red_key)