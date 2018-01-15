from setuptools import setup, Distribution
import platform


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


if platform.system() == 'Windows':
    libname = 'libmcx.dll'
elif platform.system() == 'Darwin':
    libname = 'libmcx.dylib'
else:
    libname = 'libmcx.so'


setup(
    name='pymcx',
    version='0.1',
    description='MCX Library',
    author='Qianqian Fang',
    url='https://github.com/fangq/mcx',
    python_requires='>=3',
    install_requires=['numpy'],
    packages=['pymcx'],
    package_data={
        'pymcx': [libname],
    },
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution
)
