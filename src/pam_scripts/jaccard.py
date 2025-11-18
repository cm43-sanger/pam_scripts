import numpy as np
from collections.abc import Sequence
from . import _jaccard


def _all_arrays_are_valid(arrays):
    return all(
        isinstance(array, np.ndarray) and array.dtype == np.uint64 and array.ndim == 1
        for array in arrays
    )


def get_pairwise_distances(
    arrays: Sequence[np.ndarray[tuple[int], np.dtype[np.uint64]]],
    kmer_length: int,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    if not _all_arrays_are_valid(arrays):
        raise ValueError("all arrays must be 1D numpy arrays with dtype=np.uint64.")
    return _jaccard.get_pairwise_distances(arrays, kmer_length)


def get_distances(
    reference_arrays: Sequence[np.ndarray[tuple[int], np.dtype[np.uint64]]],
    query_arrays: Sequence[np.ndarray[tuple[int], np.dtype[np.uint64]]],
    kmer_length: int,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    if not (
        _all_arrays_are_valid(reference_arrays) and _all_arrays_are_valid(query_arrays)
    ):
        raise ValueError("all arrays must be 1D numpy arrays with dtype=np.uint64.")
    return _jaccard.get_distances(reference_arrays, query_arrays, kmer_length)
