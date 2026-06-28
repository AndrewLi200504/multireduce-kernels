from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="multireduce_kernels",
    ext_modules=[
        CUDAExtension(
            name="multireduce_kernels",
            sources=["reduction_binding.cpp", "reduction.cu"],
            extra_compile_args={
                "nvcc": ["-arch=sm_86"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)