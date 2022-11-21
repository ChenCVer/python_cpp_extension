#!/usr/bin/env python
import os
import sys
import glob
import time
import torch
import warnings
import subprocess
from setuptools import find_packages, setup
from torch.utils.cpp_extension import include_paths, library_paths
from torch.utils.cpp_extension import (BuildExtension, CppExtension, CUDAExtension)


def parse_requirements(fname='requirements.txt', with_version=True):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def make_extension(name, module, sources, sources_cuda=[], use_cpp_opencv=False):
    """
    make extension modules
    """
    include_dirs = []
    library_dirs = []
    libraries = []
    define_macros = []
    extra_compile_args = {'cxx': []}

    # use 3rdparty: opencv
    if use_cpp_opencv:
        # pybind11
        include_dirs.append(os.path.abspath('./3rdparty/pybind11/include'))
        # opencv
        if sys.platform == 'win32':
            # include_directories()
            include_dirs.append(os.path.abspath('./3rdparty/opencv/win/include'))
            # target_link_directories()
            library_dirs.append(os.path.abspath('./3rdparty/opencv/win/x64/vc15/lib'))
            # target_link_libraries()
            libraries.append('opencv_world453')
        elif sys.platform == "linux":
            # include_directories()
            include_dirs.append(os.path.abspath('./3rdparty/opencv/linux/include'))
            # target_link_directories()
            library_dirs.append(os.path.abspath('./3rdparty/opencv/linux/lib'))
            # target_link_libraries()
            libraries.append("opencv_world")
        else:
            raise ValueError(f'Unsupported platform: {sys.platform}')

    # gpu files
    if torch.cuda.is_available():
        print(f'Compiling {name} with CUDA')
        include_dirs += include_paths(cuda=True)
        library_dirs += library_paths(cuda=True)
        libraries.append('cudart')
        # add macros WITH_CUDA in cpp code.
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} with cpp, but without CUDA')
        extension = CppExtension

    ext_ops = extension(
                name=f'{module}.{name}',
                sources=[os.path.join(*module.split('.'), p) for p in sources],
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries)

    return ext_ops


if __name__ == '__main__':
    setup(
        name='orbbec',
        version="0.0.1",
        description='C++/CUDA extension for python or pytorch',
        author='orbbec jiaozhu',
        author_email='jiaozhu@orbbec.com',
        keywords='C++/CUDA, python, pytorch',
        url='https://github.com/ChenCVer/PYTHON_CPP_EXTENSION',
        packages=find_packages(exclude=('3rdparty', 'requirements', 'scripts', "tools")),
        package_data={'orbbec': ['*/*.pyd']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        ext_modules=[   
            # compile info
            make_extension(
                name='compiling_info',
                module='orbbec.utils',
                sources=['src/compiling_info.cpp']),
            
            # nms    
            make_extension(
                name='nms_ext',
                module='orbbec.nms',
                sources=[
                    'src/nms_ext.cpp', 
                    'src/cpu/nms_cpu.cpp'],
                sources_cuda=[
                    'src/cuda/nms_cuda.cpp', 
                    'src/cuda/nms_kernel.cu']),
            
            # roi align    
            make_extension(
                name='roi_align_ext',
                module='orbbec.roi_align',
                sources=[
                    'src/roi_align_ext.cpp',
                    'src/cpu/roi_align_v2.cpp'],
                sources_cuda=[
                    'src/cuda/roi_align_kernel.cu',
                    'src/cuda/roi_align_kernel_v2.cu']),
   
            # warpaffine    
            make_extension(
                name='warpaffine_ext',
                module='orbbec.warpaffine',
                sources=[
                    'src/warpaffine_ext.cpp',                  
                    'src/cpu/warpaffine_opencv.cpp',
                    'src/cpu/warpaffine_torch_v2.cpp'],
                sources_cuda=[
                    'src/cuda/warpaffine_cuda.cu',
                    'src/cuda/warpaffine_kernel.cpp'],
                use_cpp_opencv=True),                      
        ],
        # if use_ninja is True, it will speed up the compilation process.
        cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},
        zip_safe=False)