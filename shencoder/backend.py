import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

_backend = load(name='_sh_encoder',
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3'],
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'shencoder.cu',
                    'bindings.cpp',
                ]],
                )

__all__ = ['_backend']