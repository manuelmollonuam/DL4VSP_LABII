# This file contains the cffi-extension call to build the custom
# kernel used by amp.
# For mysterious reasons, it needs to live at the top-level directory.
# TODO: remove this when we move to cpp-extension.


import os
import torch
from torch.utils.ffi import create_extension

abs_path = os.path.dirname(os.path.realpath(__file__))

sources = ['apex/amp/src/scale_cuda.c']
headers = ['apex/amp/src/scale_cuda.h']
defines = [('WITH_CUDA', None)]
with_cuda = True

extra_objects = [os.path.join(abs_path, 'build/scale_kernel.o')]

# When running `python build_cffi.py` directly, set package=False. But
# if it's used with `cffi_modules` in setup.py, then set package=True.
package = (__name__ != '__main__')

extension = create_extension(
    'apex.amp._C.scale_lib',
    package=package,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    extension.build()
