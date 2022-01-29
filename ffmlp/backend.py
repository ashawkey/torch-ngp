import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

_backend = load(name='_ffmlp',
                extra_cflags=['-O3'], # '-std=c++17'
                extra_cuda_cflags=[
                    '-O3',
                    '-Xcompiler=-mf16c', '-Xcompiler=-Wno-float-conversion', '-Xcompiler=-fno-strict-aliasing', '--extended-lambda', '--expt-relaxed-constexpr',
                    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', # undefine flags, necessary!
                ],
                extra_include_paths=[
                    os.path.join(_src_path, 'include'),
                ],
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'ffmlp.cu',
                    'bindings.cpp',
                ]],
                )

__all__ = ['_backend']