from . import _kmc

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


def load_kmers(filename: str, num_threads: typing.Optional[int] = None):
    with NamedTemporaryFile() as kmer_file:
        _kmc.call_kmc_tools(
            ["transform", filename, "-ci1", "dump", "-s", kmer_file.name],
            num_threads=num_threads,
        )
        with open(kmer_file.name) as f:
            uncompressed_kmers = (line.strip().split("\t", maxsplit=1)[0] for line in f)
            return np.fromiter(compress_kmers(uncompressed_kmers), np.uint64)
