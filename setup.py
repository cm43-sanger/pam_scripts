from setuptools import setup, Extension
import pybind11
import sys

extra_compile_args = ["-O3", "-std=c++17"]
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17"]

ext_modules = [
    Extension(
        "pam_scripts._kmers",
        sources=[
            "src/pam_scripts/_kmers.cpp",
            "src/pam_scripts/kmc_api/kmc_file.cpp",
            "src/pam_scripts/kmc_api/kmer_api.cpp",
            "src/pam_scripts/kmc_api/mmer.cpp",
        ],
        include_dirs=[pybind11.get_include(), "src/pam_scripts/kmc_api"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    ext_modules=ext_modules,
)
