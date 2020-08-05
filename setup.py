import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from json import loads
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def _check_cmake(major: int, minor: int) -> bool:
    result = subprocess.run(
        ["cmake", "-E", "capabilities"], capture_output=True, text=True
    )
    if result.returncode != 0:
        msg = f"Failed to get CMake capabilities with error code {result.returncode} and error"
        print(msg, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return True
    ver = loads(result.stdout)["version"]
    ver = (ver["major"], ver["minor"])
    if ver < (major, minor):
        msg = f"CMake version {ver} is below minimum requirement of ({major}, {minor})"
        print(msg, file=sys.stderr)
        return True
    return False


class CMakeExtension(Extension):
    def __init__(self, name, target=None, **kwargs):
        super().__init__(name, sources=[], **kwargs)
        if target is None:
            self.target = name.split(".")[-1]
        else:
            self.target = target


class CMakeBuildExt(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_type = "Debug" if self.debug else "Release"
        target_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        subprocess.check_call(
            [
                "cmake",
                "-S",
                ".",
                "-B",
                self.build_temp,
                f"-DCMAKE_BUILD_TYPE={build_type}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={target_path.parent}",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DPython3_EXECUTABLE={sys.executable}",
            ]
        )
        subprocess.check_call(
            [
                "cmake",
                "--build",
                self.build_temp,
                "--clean-first",
                "--target",
                ext.target,
            ],
        )


setup_requires = ["setuptools_scm", "cffi>=1.14.2"]
if _check_cmake(3, 16):
    setup_requires.append("cmake>=3.16")

setup(
    setup_requires=setup_requires,
    ext_modules=[CMakeExtension("pymcx._pymcx", py_limited_api=True)],
    cmdclass=dict(build_ext=CMakeBuildExt),
    packages=["pymcx"],
    package_dir={"": "python"},
    include_package_data=False,
    # use_scm_version=True,
    # TODO get these fields from pyproject.toml
    name="pymcx",
    version="0.4.7",
    description="Python bindins to Monte Carlo eXtreme (MCX)",
    long_description="README.md",
    python_requires=">=3.6",
    license="LICENSE.txt",
    author="Qianqian Fang",
    author_email="q.fang@neu.edu",
    url="http://github.com/fangq/mcx",
    install_requires=[
        "cffi>=1.14.2",
        "numpy",
        "typing_extensions>=3.7.4.3",
    ],
)
