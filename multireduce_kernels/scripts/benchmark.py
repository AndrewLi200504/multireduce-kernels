from argparse import ArgumentParser
import torch
import multireduce_kernels as mk
import inspect
import json
import torch_helpers as th
import mk_workloads as mkw

from typing import Any, Callable
from contextlib import contextmanager
from time import perf_counter


def cast_if_long(mk_ret):
    return mk_ret.to(torch.long) if mk_ret.dtype is torch.uint64 else mk_ret
def compute_accuracy(torch_ret, mk_ret, rtol=1e-5, atol=1e-5):
    if torch_ret.dtype != mk_ret.dtype:
        mk_ret: torch.tensor = mk_ret.to(torch_ret.dtype)
    return int(
        torch.allclose(
            torch_ret,
            mk_ret,
            rtol=rtol,
            atol=atol
        )
    )


class TorchReduction(): 
    tuple_idx: int | list
    reduction: Callable[..., Any]
    timestep: float
    name: str
    def args_filterer(func):
      
        def wrapper(self, *args):
            if isinstance(self.tuple_idx, list):
                args = tuple(args[i] for i in self.tuple_idx)
            elif self.tuple_idx != -1:
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
        tr = cls()
        if hasattr(th, fn_key):
            tr.reduction = getattr(th, fn_key)
        elif fn_key in th.SPECIAL_REDS:
            tr.reduction = th.SPECIAL_REDS[fn_key] 
        else:
            tr.reduction = getattr(torch, fn_key)
        tr.tuple_idx = _dict[fn_key]
        tr.name = tr.reduction.__name__
        return tr

    @classmethod
    def from_string(cls, string):
        self = cls()
        self.reduction = th.SPECIAL_REDS[string] if string in th.SPECIAL_REDS else getattr(th, string)
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
    non_tens_args: tuple | None
    def init_float(self): 
        return torch.randn(self.tens_len, device="cuda")
    def init_bool(self):
        return torch.randint(0, 2, (1, self.tens_len), dtype=bool, device="cuda")
    def init_prob(self):
        prob_dist = torch.rand(self.tens_len, device="cuda")
        return prob_dist / prob_dist.sum()
    def init_float_debug(self):
        original = torch.randn(self.tens_len, device="cuda")
        r = torch.rand(self.tens_len)
        original[r < 0.1] = float('nan')
        original[(r >= 0.1) & (r < 0.2)] = float('inf')
        original[(r >= 0.2) & (r < 0.3)] = float('-inf')
        return original
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
                   tens_init_fn=cls.init_float,
                   non_tens_args=None)
    
    def __init__(self, str: str, tens_len: int, num_it: int):
        self.torch_total_time = 0
        self.mk_total_time = 0
        self.torch_reds = []
        self.non_tens_args = None
        if hasattr(mk, str): 
            self.mk_red = getattr(mk, str)
            reduction_dict: dict = json.loads(self.mk_red.__doc__.split('|')[1])
            self.tens_init_fn = getattr(self, f"init_{reduction_dict['type']}")
            self.num_tens_args = reduction_dict["args"]
            for torch_red_dict in reduction_dict["reds"]:
                self.torch_reds.append(TorchReduction.from_dict(torch_red_dict))
            extra_args = reduction_dict.get("extra_args")
            if extra_args is not None: 
                tuplized_extra_args = []
                for extra_arg in extra_args:
                    if isinstance(extra_arg, list):
                        tuplized_extra_args.append(list(extra_arg))
                    else:
                        tuplized_extra_args.append(extra_arg)
            self.non_tens_args = tuple(tuplized_extra_args) if extra_args is not None else None
                        
        else:
            self.mk_red = getattr(mkw, str)
            if hasattr(self.mk_red, "init_type_name"):
                self.tens_init_fn = getattr(self, f"init_{self.mk_red.init_type_name}")
            else: 
                print("Invalid benchmark")
            self.num_tens_args = len(inspect.signature(self.mk_red).parameters)
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
        args_tuple = tens_tuple + (self.non_tens_args or ())            
        torch_rets = []
        mk_tuple = None
        for i, torch_red in enumerate(self.torch_reds):
            
            ret = torch_red(*args_tuple)
            torch_rets.append(ret)
            self.torch_total_time += torch_red.get_timestep()
        with timeit_and_synch(self): 
            mk_tuple = self.mk_red(*args_tuple)
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


@contextmanager
def timeit_and_synch(cls: TorchReduction | Benchmark):
    torch.cuda.synchronize()

    start = perf_counter()
    yield
    torch.cuda.synchronize()
    end = perf_counter()
    cls.timestep = end - start
