from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import pybind11
import sys

# --- Compile flags ---
extra_compile_args_cpp = ["-O3", "-std=c++17"]
extra_compile_args_cython = ["-O3"]

if sys.platform == "win32":
    extra_compile_args_cpp = ["/O2", "/std:c++17"]
    extra_compile_args_cython = ["/O2"]

# --- C++ pybind11 extension ---
kmers_ext = Extension(
    "pam_scripts._kmers",
    sources=[
        "src/pam_scripts/_kmers.cpp",
        "src/pam_scripts/kmc_api/kmc_file.cpp",
        "src/pam_scripts/kmc_api/kmer_api.cpp",
        "src/pam_scripts/kmc_api/mmer.cpp",
    ],
    include_dirs=[pybind11.get_include(), "src/pam_scripts/kmc_api"],
    language="c++",
    extra_compile_args=extra_compile_args_cpp,
)

# --- Cython extension ---
jaccard_ext = Extension(
    "pam_scripts._jaccard_similarity",
    sources=["src/pam_scripts/jaccard_similarity.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args_cython,
)

setup(
    name="pam_scripts",
    ext_modules=cythonize(
        [jaccard_ext],  # only Cython modules go through cythonize
        compiler_directives={"language_level": "3"},
        annotate=True,
    )
    + [kmers_ext],  # add the C++ extensions directly
)
