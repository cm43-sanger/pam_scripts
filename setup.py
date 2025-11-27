from setuptools import setup, Extension
import pybind11
import sys


if sys.platform == "win32":
    extra_compile_args_cpp = ["/O2", "/std:c++17"]
else:
    extra_compile_args_cpp = ["-O3", "-std=c++17", "-march=native"]

kc_ext = Extension(
    "pam_scripts._kmc",
    sources=[
        "src/pam_scripts/_kmc.cpp",
        "src/pam_scripts/kmc_api/kmc_file.cpp",
        "src/pam_scripts/kmc_api/kmer_api.cpp",
        "src/pam_scripts/kmc_api/mmer.cpp",
    ],
    include_dirs=[pybind11.get_include(), "src/pam_scripts/kmc_api"],
    language="c++",
    extra_compile_args=extra_compile_args_cpp,
)
hash_ext = Extension(
    "pam_scripts._xxhash",
    sources=["src/pam_scripts/_xxhash.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=extra_compile_args_cpp,
)
jaccard_ext = Extension(
    "pam_scripts._jaccard",
    sources=["src/pam_scripts/_jaccard.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=extra_compile_args_cpp,
)

setup(name="pam_scripts", ext_modules=[kc_ext, hash_ext, jaccard_ext])
