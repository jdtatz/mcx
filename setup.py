#!/usr/bin/env python3
from skbuild import setup

setup(
    name='pymcx',
    version='0.2',
    description='MCX Library',
    author='Qianqian Fang',
    url='https://github.com/fangq/mcx',
    setup_requires=["scikit-build", "cmake", "cython"],
    install_requires=['numpy'],
    include_package_data=False,
    zip_safe=False,
    python_requires=">=3.6",
    packages=['pymcx'],
    cmake_languages=("C", "CXX", "CUDA"),
    cmake_minimum_required_version="3.13",
)
