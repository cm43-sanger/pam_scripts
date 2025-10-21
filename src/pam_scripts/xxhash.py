from . import _xxhash

import numpy as np
import typing

UINT64_MAX = 2**64 - 1


def hash_kmers(
    kmers: np.ndarray[tuple[int], np.dtype[np.uint64]],
    seed: int = 42,
    num_threads: typing.Optional[int] = None,
) -> np.ndarray[tuple[int], np.dtype[np.uint64]]:
    num_threads = num_threads or 1
    if seed < 1 or seed > UINT64_MAX:
        raise ValueError("seed must be in range [1, 2^64-1]")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    return _xxhash.hash_kmers(kmers, seed=seed, num_threads=num_threads)
