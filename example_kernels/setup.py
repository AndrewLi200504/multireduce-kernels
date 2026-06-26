from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="example_kernels",
    ext_modules=[
        CUDAExtension(
            name="example_kernels",
            sources=["max_binding.cpp", "max.cu"],
            extra_compile_args={
                "nvcc": ["-arch=sm_86"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)