from benchmark import Benchmark
import multireduce_kernels as mk
import mk_workloads as mkw


def benchmark_one_fn(name, f):
    if callable(f): 
        print(f"Benchmarking {name}")
        benchmark = Benchmark(name, 10000000, 100)
        failures = benchmark()
        print("______________________________________")
        return failures
    else:
        return None


if __name__ == "__main__": 
    failures = []
    benchmarks = 0
    for name, f in mk.__dict__.items(): 
        failure = benchmark_one_fn(name, f)
        #failures += benchmark_one_fn(name, f)
        if failure is None:
            continue
        failures += failure
        benchmarks += 1
    for name, f in mkw.__dict__.items():
        if not hasattr(f, "init_type_name"):
            continue
        failure = benchmark_one_fn(name, f)
        if failure is None:
            continue
        failures += failure
        benchmarks += 1
    print("\033[32m" + f"Passed: {benchmarks - len(failures)} / {benchmarks}" + "\033[0m")
    for red_name, red_accuracy in failures:
        print("\033[31m" + f"Fail: {red_name} due to accuracy {red_accuracy}%" + "\033[0m")