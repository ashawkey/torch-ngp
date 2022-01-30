import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

_backend = load(name='_raymarching',
                extra_cflags=['-O3'], # '-std=c++17'
                extra_cuda_cflags=['-O3'], # '-arch=sm_70'
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'raymarching.cu',
                    'bindings.cpp',
                ]],
                )

__all__ = ['_backend']