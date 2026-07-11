## Multireduce Kernels

This project aims to perform multiple reductions over vectors to avoid redundant passes over the same vector. For example, in standard torch, in order to get the min and the max with `torch.max()` and `torch.min()`, you have to pass over the same input tensor twice. This makes a simultaneous reduction operation (calculate both min and max of a tensor, `t`) that should take roughly the same amount of time as simply using one of the individual kernels (just min or just max of `t`) take twice as long to run. 

## Instructions to run

Ensure you have an NVIDIA CUDA GPU. 
- `cd multireduce_kernels`
- `python -m venv venv`
- `source venv/bin/activate`
- `pip install torch --index-url https://download.pytorch.org/whl/YOUR_WHL_HERE` 
- `pip install setuptools`
- `pip install -e . --no-build-isolation`
