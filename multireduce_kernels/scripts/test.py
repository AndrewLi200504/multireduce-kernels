from argparse import ArgumentParser
import torch
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
Reduction = namedtuple('Reduction', ['red0', 'red1', 'fusedred'])
fn_dict: dict[str, Reduction] = {
    "min_max": Reduction(red0=torch.min, red1=torch.max, fusedred=mk.min_max),
    "min_argmin": Reduction(red0=torch.min, red1=torch.argmin, fusedred=mk.min_argmin),
    "max_argmax": Reduction(red0=torch.max, red1=torch.argmax, fusedred=mk.max_argmax),
    "sum_sumsq": Reduction(red0=torch.sum, red1=sumsq, fusedred=mk.sum_sumsq)
}


def main(red_key):

    torch_total_time = 0
    mk_total_time = 0
    inacc_red0 = 0
    inacc_red1 = 0
    red = fn_dict[red_key]
    for _ in range(100):
        t = torch.randn(1000000).to("cuda")
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
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("reduction", help="Fused reduction to use") 
    args = parser.parse_args()
    main(args.reduction)