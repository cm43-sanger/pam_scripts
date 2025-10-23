import numpy as np
from collections.abc import Sequence
from . import _jaccard


def get_pairwise_jaccard_distances(
    arrays: Sequence[np.ndarray[tuple[int], np.dtype[np.uint64]]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    if not all(
        isinstance(a, np.ndarray) and a.dtype == np.uint64 and a.ndim == 1
        for a in arrays
    ):
        raise ValueError("All arrays must be 1D numpy arrays with dtype=np.uint64.")
    return _jaccard.get_pairwise_jaccard_distances(arrays)
