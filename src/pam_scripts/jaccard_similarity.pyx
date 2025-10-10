# jaccard_similarity.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3

import numpy as np
cimport numpy as np
from cython.parallel import prange

ctypedef np.uint64_t UINT64_t


# ----------------------------------------------------------------------
# Internal intersection helper (nogil)
# ----------------------------------------------------------------------
cdef inline Py_ssize_t intersect_size(const UINT64_t[:] a, Py_ssize_t len_a,
                                      const UINT64_t[:] b, Py_ssize_t len_b) nogil:
    """
    Count the intersection size between two sorted uint64 arrays using two pointers.
    This function runs without the GIL and does no Python-level operations.
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


# ----------------------------------------------------------------------
# Internal Jaccard implementation (nogil)
# ----------------------------------------------------------------------
cdef inline double _jaccard_similarity(const UINT64_t[:] a, const UINT64_t[:] b) nogil:
    """
    Internal nogil-safe implementation that returns the Jaccard similarity
    between two sorted, unique uint64 arrays.
    """
    cdef Py_ssize_t len_a = a.shape[0]
    cdef Py_ssize_t len_b = b.shape[0]
    cdef Py_ssize_t inter = intersect_size(a, len_a, b, len_b)
    cdef Py_ssize_t union_size = len_a + len_b - inter
    if union_size == 0:
        return 0.0
    return (<double>inter) / (<double>union_size)


# ----------------------------------------------------------------------
# Public: Jaccard similarity for a single pair
# ----------------------------------------------------------------------
cpdef double jaccard_similarity(const UINT64_t[:] a, const UINT64_t[:] b):
    """
    Compute Jaccard similarity between two sorted, unique uint64 arrays.
    Calls the internal nogil implementation.
    """
    cdef double res
    with nogil:
        res = _jaccard_similarity(a, b)
    return res


# ----------------------------------------------------------------------
# Public: Parallel pairwise Jaccard similarities
# ----------------------------------------------------------------------
cpdef np.ndarray pairwise_jaccard(tuple arrays):
    """
    Compute pairwise Jaccard similarities between a tuple of sorted uint64 arrays.
    Uses OpenMP parallelism for speed.
    Returns an n x n symmetric numpy array (dtype=float64).
    """
    cdef Py_ssize_t n = len(arrays)
    cdef np.ndarray[np.float64_t, ndim=2] result = np.empty((n, n), dtype=np.float64)
    cdef Py_ssize_t i, j

    cdef np.ndarray[np.uint64_t, ndim=1] ai, bj
    cdef const UINT64_t[:] ai_view, bj_view

    # Fill diagonal
    for i in range(n):
        result[i, i] = 1.0

    # Parallelize the upper triangle
    with nogil:
        for i in prange(1, n, schedule='dynamic'):
            ai = arrays[i]
            ai_view = ai
            for j in range(i):
                bj = arrays[j]
                bj_view = bj
                result[i, j] = _jaccard_similarity(ai_view, bj_view)
                result[j, i] = result[i, j]

    return result
