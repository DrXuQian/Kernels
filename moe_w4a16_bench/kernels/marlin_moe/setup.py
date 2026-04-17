from setuptools import setup
from torch.utils import cpp_extension
import glob

# Compile all SM80 kernel variants
import glob
kernel_files = sorted(glob.glob('sm80_kernel_*.cu'))

setup(
    name='marlin_moe_wna16',
    ext_modules=[cpp_extension.CUDAExtension(
        'marlin_moe_cuda',
        ['binding.cpp', 'ops.cu'] + kernel_files,
        include_dirs=['.', 'deps'],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--expt-relaxed-constexpr',
                     '-DMARLIN_NAMESPACE_NAME=marlin_moe_wna16',
                     '-DTORCH_EXTENSION_NAME=marlin_moe_cuda'],
        },
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
