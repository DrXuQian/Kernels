from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='moe_wna16',
    ext_modules=[cpp_extension.CUDAExtension(
        'moe_wna16_cuda',
        ['moe_wna16_binding.cpp', 'moe_wna16.cu'],
        extra_compile_args={'nvcc': ['-O3', '--expt-relaxed-constexpr']},
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
