from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import multireduce_kernels as mk

from collections import namedtuple
from contextlib import contextmanager
from time import perf_counter

@contextmanager
def timeit():
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"Elapsed time: {end - start:.2f} seconds")

def sumsq(t):
    return torch.sum(t ** 2)
def cos_sim(t0, t1): 
    return F.cosine_similarity(t0, t1, dim=0)
def l2_norm(t0, t1):
    return torch.linalg.vector_norm(t0 - t1)
def mk_cos_sim(asq, ab, bsq):
    return ab / (asq.sqrt() * bsq.sqrt())
def mk_l2_norm(asq, ab, bsq): 
    return torch.sqrt(asq - 2 * ab + bsq)
 
DoubleReduction = namedtuple('DoubleReduction', ['red0', 'red1', 'fusedred'])
double_red_dict: dict[str, DoubleReduction] = {
    "min_max": DoubleReduction(red0=torch.min, red1=torch.max, fusedred=mk.min_max),
    "min_argmin": DoubleReduction(red0=torch.min, red1=torch.argmin, fusedred=mk.min_argmin),
    "max_argmax": DoubleReduction(red0=torch.max, red1=torch.argmax, fusedred=mk.max_argmax),
    "sum_sumsq": DoubleReduction(red0=torch.sum, red1=sumsq, fusedred=mk.sum_sumsq)
}

TripleReduction = namedtuple('TripleReduction', ['original', 'fusedred', 'mk_reducer'])
triple_red_dict: dict[str, TripleReduction] = {
    "cosine_similarity": TripleReduction(original=cos_sim, fusedred=mk.asq_ab_bsq, mk_reducer=mk_cos_sim),
    "l2_norm": TripleReduction(original=l2_norm, fusedred=mk.asq_ab_bsq, mk_reducer=mk_l2_norm)
}
def double_red(red_key):

    torch_total_time = 0
    mk_total_time = 0
    inacc_red0 = 0
    inacc_red1 = 0
    red = double_red_dict[red_key]
    for _ in range(100):
        t = torch.randn(10000000).to("cuda")
        torch.cuda.synchronize()
        start_torch = perf_counter()
        red0_torch = red.red0(t)
        red1_torch = red.red1(t)
        torch.cuda.synchronize()
        end_torch = perf_counter()
        torch_total_time += (end_torch - start_torch)
        torch.cuda.synchronize()
        start_mk = perf_counter()
        red0_mk, red1_mk = red.fusedred(t)
        torch.cuda.synchronize()
        end_mk = perf_counter()
        mk_total_time += (end_mk - start_mk)
        if abs(red0_mk - red0_torch) < 1e-2:
            inacc_red0 += 1
        red1_mk = red1_mk.to(torch.long) if red1_mk.dtype is torch.uint64 else red1_mk

        if abs(red1_mk - red1_torch) < 1e-2:
            inacc_red1 += 1
    torch_avg_time = torch_total_time / 100
    mk_avg_time = mk_total_time / 100
    red0_accuracy = inacc_red0 / 100
    red1_accuracy = inacc_red1 / 100
    print(f"Average time on torch: {torch_avg_time}")
    print(f"Average time on mk: {mk_avg_time}")
    print(f"{red.red0.__name__} accuracy: {red0_accuracy}")
    print(f"{red.red1.__name__} accuracy: {red1_accuracy}")
def triple_red(red_key): 
    
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
        triple_red_mk = red.mk_reducer(*red.fusedred(t0, t1))
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
    parser.add_argument("red_key", help="Fused double_red_dict to use") 
    args = parser.parse_args()
    if args.red_key in double_red_dict: 
        double_red(args.red_key)
    elif args.red_key in triple_red_dict:
        triple_red(args.red_key)