# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import glob
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

def cuda_setup():
    if not torch.cuda.is_available():
        print("CUDA not available, building CPU version only.")
        return False
    else:
        print("CUDA available, building with GPU support.")
        return True

if cuda_setup():
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/{}/include".format(ROOT, _ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/{}/include".format(ROOT, _ext_src_root))],
            },
        )
    ],
else:
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            # include_dirs=["{}/{}/include".format(ROOT, _ext_src_root)],
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/{}/include".format(ROOT, _ext_src_root))],
            },
        )
    ]


setup(
    name='pointnet2',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)