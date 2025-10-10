# jaccard_similarity.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np

ctypedef np.uint64_t UINT64_t

cdef inline Py_ssize_t intersect_size(const UINT64_t[:] a, Py_ssize_t len_a,
                                      const UINT64_t[:] b, Py_ssize_t len_b) nogil:
    """
    Two-pointer intersection count. Assumes `a` and `b` are sorted.
    This function is nogil-safe and does NOT access Python-level attributes.
    """
    cdef Py_ssize_t i = 0, j = 0, intersection = 0
    cdef UINT64_t ai, bj
    while i < len_a and j < len_b:
        ai = a[i]
        bj = b[j]
        intersection += (ai == bj)
        i += (ai <= bj)
        j += (ai >= bj)
    return intersection


cpdef double jaccard_similarity(const UINT64_t[:] a, const UINT64_t[:] b):
    """
    Compute Jaccard similarity between two sorted, unique uint64 arrays.
    Accepts numpy arrays (or any buffer supporting the memoryview protocol).
    Returns a floating-point value in [0.0, 1.0].
    """
    cdef Py_ssize_t len_a = a.shape[0]
    cdef Py_ssize_t len_b = b.shape[0]
    cdef Py_ssize_t inter, union_size

    # call the nogil inner loop (safe)
    with nogil:
        inter = intersect_size(a, len_a, b, len_b)

    union_size = len_a + len_b - inter
    if union_size == 0:
        return 0.0
    # ensure floating-point division:
    return (<double>inter) / (<double>union_size)