#!/usr/bin/env python3
from skbuild import setup
import platform

if platform.system() == 'Windows':
    libname = 'libmcx.dll'
elif platform.system() == 'Darwin':
    libname = 'libmcx.dylib'
else:
    libname = 'libmcx.so'


setup(
    name='pymcx',
    version='0.1.1',
    description='MCX Library',
    author='Qianqian Fang',
    url='https://github.com/fangq/mcx',
    install_requires=['numpy'],
    packages=['pymcx'],
    package_data={'pymcx': [libname]},
    include_package_data=True,
    zip_safe=False,
    cmake_languages=("C", "CXX", "CUDA"),
)
