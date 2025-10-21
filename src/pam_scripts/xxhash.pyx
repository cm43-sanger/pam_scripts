# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from libc.stdint cimport uint64_t
import numpy as np
cimport numpy as np

cdef extern from "xxhash.h":
    ctypedef unsigned long long XXH64_hash_t
    XXH64_hash_t XXH64(const void* input, size_t length, unsigned long long seed)

# Declare the numpy dtype for output
DTYPE = np.uint64

def xxh64(np.ndarray arr, unsigned long long seed=0):
    """
    Compute XXH64 hash for each element in an object array of bytes.
    Returns a uint64 numpy array of hashes.
    """
    cdef Py_ssize_t n = arr.size
    cdef np.ndarray[np.uint64_t, ndim=1] out = np.empty(n, dtype=DTYPE)
    cdef Py_ssize_t i
    cdef object item
    cdef const char *buf
    cdef Py_ssize_t size
    cdef XXH64_hash_t h

    if arr.dtype != np.object_:
        raise TypeError("Input must be a NumPy object array of bytes")

    for i in range(n):
        item = arr[i]
        if not isinstance(item, (bytes, bytearray)):
            raise TypeError(f"Element {i} is not bytes")
        size = len(item)
        buf = item  # Cython automatically gets pointer from bytes
        h = XXH64(buf, size, seed)
        out[i] = h

    return out
