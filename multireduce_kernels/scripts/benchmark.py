from argparse import ArgumentParser
import torch
import multireduce_kernels as mk
import inspect
import json
from torch_helpers import SPECIAL_REDS
import mk_workloads as mkw

from typing import Any, Callable
from contextlib import contextmanager
from time import perf_counter


def cast_if_long(mk_ret):
    return mk_ret.to(torch.long) if mk_ret.dtype is torch.uint64 else mk_ret
def compute_accuracy(torch_ret, mk_ret):
    return 1 if abs(mk_ret - torch_ret) < 1e-2 else 0

@contextmanager
def timeit_and_synch(cls):
    torch.cuda.synchronize()

    start = perf_counter()
    yield
    torch.cuda.synchronize()
    end = perf_counter()
    cls.timestep = end - start


class TorchReduction(): 
    tuple_idx: int
    reduction: Callable[..., Any]
    timestep: float
    name: str
    def args_filterer(func):
      
        def wrapper(self, *args):
            if self.tuple_idx != -1:
                i = self.tuple_idx
                args = args[i: i + 1]
            result = func(self, *args)
            return result
       
                
        return wrapper
    @args_filterer
    def __call__(self, *args):
        with timeit_and_synch(self):
            return self.reduction(*args) 
        
    @classmethod
    def from_dict(cls, _dict):
        fn_key = next(iter(_dict))
        self = cls()
        self.reduction = SPECIAL_REDS[fn_key] if fn_key in SPECIAL_REDS else getattr(torch, fn_key)
        self.tuple_idx = _dict[fn_key]
        self.name = self.reduction.__name__
        return self

    @classmethod
    def from_string(cls, string):
        self = cls()
        self.reduction = SPECIAL_REDS[string]
        self.tuple_idx = -1
        self.name = string 
        return self
    def get_timestep(self):
        return self.timestep
    def get_name(self):
        return self.name
class Benchmark:
    torch_total_time: int
    mk_total_time: int
    torch_reds: list[TorchReduction]
    mk_red: Callable[..., tuple]
    acc_count: list
    tens_len: int
    num_tens_args: int  
    timestep: float
    num_it: int
    tens_init_fn: Callable[..., Any]
    def init_float(self): 
        return torch.randn(self.tens_len, device="cuda")
    def init_bool(self):
        return torch.randint(0, 2, (1, self.tens_len), dtype=bool, device="cuda")
  
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
                   num_it=0, 
                   tens_init_fn=cls.init_float)
    
    def __init__(self, str: str, tens_len: int, num_it: int):
        self.torch_total_time = 0
        self.mk_total_time = 0
        self.torch_reds = []

        if hasattr(mk, str): 
            self.mk_red = getattr(mk, str)
            reduction_dict = json.loads(self.mk_red.__doc__)
            self.tens_init_fn = getattr(self, f"init_{reduction_dict['type']}")
            self.num_tens_args = reduction_dict["args"]
            for torch_red_dict in reduction_dict["reds"]:
                self.torch_reds.append(TorchReduction.from_dict(torch_red_dict))
        else:
            self.mk_red = getattr(mkw, str)
            if hasattr(self.mk_red, "init_type_name"):
                self.tens_init_fn = getattr(self, f"init_{self.mk_red.init_type_name}")
            else: 
                print("Invalid benchmark")
            self.num_tens_args = len(inspect.signature(self.mk_red).parameters)
            # breakpoint()
            self.torch_reds.append(TorchReduction.from_string(str))
        
        self.acc_count = [0] * len(self.torch_reds) 
        self.tens_len = tens_len
        self.num_it = num_it


    
    def __call__(self):
        for _ in range(self.num_it):
            self.benchmark_one_it()
        torch_avg_time = self.torch_total_time / self.num_it
        mk_avg_time = self.mk_total_time / self.num_it
        print(f"Average time on torch: {torch_avg_time}")
        print(f"Average time on mk: {mk_avg_time}")
        print(f"Speedup: {torch_avg_time / mk_avg_time:.2f}x")
        failures = []
        for i, acc in enumerate(self.acc_count): 
            red_accuracy = acc / self.num_it * 100
            red_name = self.torch_reds[i].get_name()
            print(f"{red_name} accuracy: {red_accuracy:.2f}%")
            if red_accuracy < 50:
                failures.append((self.mk_red.__name__, red_name, red_accuracy))
        return failures
        
        
    def benchmark_one_it(self):
        tens_tuple = tuple(
            self.tens_init_fn()
            for _ in range(self.num_tens_args)
        )
        torch_rets = []
        mk_tuple = None
        for i, torch_red in enumerate(self.torch_reds):
            ret = torch_red(*tens_tuple)
            torch_rets.append(ret)
            self.torch_total_time += torch_red.get_timestep()
        with timeit_and_synch(self): 
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
    benchmark = Benchmark(args.kernel, int(args.size), int(args.num_it))
    benchmark()