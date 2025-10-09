from . import _kmc, _kmers

import numpy as np
import typing
from collections.abc import Iterable
from tempfile import NamedTemporaryFile

KMER_COMPRESSOR = str.maketrans("ACGT", "0123")
KMER_DECOMPRESSOR = str.maketrans("0123", "ACGT")


def compress_kmers(kmers: Iterable[str]):
    return (int(kmer.translate(KMER_COMPRESSOR), base=4) for kmer in kmers)


def decompress_kmers(kmers: Iterable[int], kmer_length: int):
    return (
        np.base_repr(kmer, base=4, padding=kmer_length).translate(KMER_DECOMPRESSOR)
        for kmer in kmers
    )


def load_kmers(
    filename: str, num_threads: typing.Optional[int] = None
) -> np.ndarray[tuple[int], np.dtype[np.uint64]]:
    return _kmers.load_kmc_kmers(filename)
