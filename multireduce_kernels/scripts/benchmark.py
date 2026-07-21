from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import multireduce_kernels as mk
import inspect
from torch_helpers import SPECIAL_REDS
import mk_workloads as mkw
from collections import namedtuple
from dataclasses import dataclass, field

from typing import Any, Callable, NamedTuple, Optional
from contextlib import contextmanager
from time import perf_counter



def cast_if_long(mk_ret):
    return mk_ret.to(torch.long) if mk_ret.dtype is torch.uint64 else mk_ret
def compute_accuracy(torch_ret, mk_ret):
    return 1 if abs(mk_ret - torch_ret) < 1e-2 else 0

class Reduction:
    torch_total_time: int
    mk_total_time: int
    torch_reds: list[Callable[..., Any]]
    mk_red: Callable[..., tuple]
    acc_count: list
    tens_len: int
    num_tens_args: int
    timestep: float
    num_it: int
    @contextmanager
    def timeit_and_synch(self):
        torch.cuda.synchronize()

        start = perf_counter()
        yield
        torch.cuda.synchronize()
        end = perf_counter()
        self.timestep = end - start
    @classmethod
    def default(cls):
        return cls(torch_total_time=0, 
                   mk_total_time=0, 
                   torch_reds=[],
                   mk_red=mk.min_max, 
                   acc_count=(),
                   tens_len=0,
                   num_tens_args=0,
                   timestep=0,
                   num_it=0)
    
    def __init__(self, str: str, tens_len: int, num_it: int):
        self.torch_total_time = 0
        self.mk_total_time = 0
        self.torch_reds = []
        split = str.split('_')
        
        for delim_str in split: 
            if delim_str in SPECIAL_REDS: 
                fn = SPECIAL_REDS[delim_str]
            else:
                fn =  getattr(torch, delim_str)
            self.torch_reds.append(fn)
        special_workload = False
        if hasattr(mkw, str):
            special_workload = True
            self.mk_red = getattr(mkw, str)
        else:
            self.mk_red = getattr(mk, str)
            
        self.acc_count = [0] * len(self.torch_reds) 
        self.tens_len = tens_len
        if special_workload: 
            self.num_tens_args = len(inspect.signature(self.mk_red).parameters)
        else:
            self.num_tens_args = 2 if len(split) > 2 else 1
        self.num_it = num_it

    
    def benchmark(self):
        tens_args_map = self.populate_args_map()
        for _ in range(self.num_it):
            self.benchmark_one_it(tens_args_map)
        torch_avg_time = self.torch_total_time / self.num_it
        mk_avg_time = self.mk_total_time / self.num_it
        print(f"Average time on torch: {torch_avg_time}")
        print(f"Average time on mk: {mk_avg_time}")
        print(f"Speedup: {torch_avg_time / mk_avg_time:.2f}x")
        for i, acc in enumerate(self.acc_count): 
            print(f"{self.torch_reds[i].__name__} accuracy: {(acc / self.num_it) * 100:.2f}%")
    def populate_args_map(self):
        tens_args_map = []
        single_red_idx = 0
        special_fns = [fn.__name__ for fn in SPECIAL_REDS.values()]
        for torch_red in self.torch_reds:
            len_args = -1
            if torch_red.__name__ in special_fns:
                len_args = len(inspect.signature(torch_red).parameters)          
            if len_args == 1: 
                if self.num_tens_args >= 2:
                    len_args = single_red_idx % 2
                    single_red_idx += 1
                else:
                    len_args = -1
            else: 
                len_args = -1
            tens_args_map.append(len_args)
        return tens_args_map
    def benchmark_one_it(self, tens_args_map: list):
        tens_tuple = tuple(
            torch.randn(self.tens_len, device="cuda")
            for _ in range(self.num_tens_args)
        )
        torch_rets = []
        mk_tuple = None
        for i, torch_red in enumerate(self.torch_reds):
            tens_idx = tens_args_map[i]
            if tens_idx < 0:
                with self.timeit_and_synch(): 
                    ret = torch_red(*tens_tuple)
            else:
                with self.timeit_and_synch(): 
                    ret = torch_red(tens_tuple[tens_idx])
            torch_rets.append(ret)
            self.torch_total_time += self.timestep
        with self.timeit_and_synch(): 
            mk_tuple = self.mk_red(*tens_tuple)
        self.mk_total_time += self.timestep
        for i, (torch_ret, mk_ret) in enumerate(zip(torch_rets, mk_tuple)):
            self.acc_count[i] += compute_accuracy(torch_ret, cast_if_long(mk_ret))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("kernel", help="Fused multireduce kernel to benchmark") 
    parser.add_argument("size", help="Size of input tensor to use in benchmark")
    parser.add_argument("num_it", help="Number of iterations to benchmark for")
    args = parser.parse_args()
    red = Reduction(args.kernel, int(args.size), int(args.num_it))
    red.benchmark()