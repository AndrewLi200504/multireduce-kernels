from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import os

torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

setup(
    name="multireduce_kernels",
    ext_modules=[
        CUDAExtension(
            name="multireduce_kernels",
            sources=["reduction_binding.cpp", "reduction.cu"],
            extra_compile_args={
                "nvcc": ["-arch=sm_86"],
            },
            extra_link_args=[
                f"-Wl,-rpath,{torch_lib}",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)