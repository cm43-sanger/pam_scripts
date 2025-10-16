import shutil

for executable in ("kmc", "kmc_tools"):
    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"Required executable '{executable}' not found in PATH."
        )

from . import _kmc

import math
import numpy as np
import os
import subprocess
import typing
from collections.abc import Sequence
from tempfile import TemporaryDirectory

MAX_THREADS = 128
NUM_CPUS = os.cpu_count() or 1
MINIMUM_COUNT = 2
CLAMP_COUNT = 65_535  # 16 bit unsigned integer maximum


def load_kmers(db: str) -> np.ndarray[tuple[int], np.dtype[np.uint64]]:
    return _kmc.load_kmers(db)


def estimate_coverage(db: str) -> float:
    return _kmc.estimate_coverage(db)


def _resolve_num_threads(num_threads: typing.Optional[int]):
    if num_threads is None:
        num_threads = NUM_CPUS
    if num_threads < 1:
        raise ValueError("number of threads must be positive")
    return min(num_threads, MAX_THREADS)


def _call_kmc(args: Sequence[str], error_message: str):
    result = subprocess.run(args, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode(errors="ignore")
        raise RuntimeError(f"\n{err}\n{error_message}.")


def _count_kmers(
    read: str,
    output_db: str,
    kmer_length: int = 21,
    num_threads: typing.Optional[int] = None,
):
    if not os.path.exists(read):
        raise FileNotFoundError(read)
    if kmer_length < 1 or kmer_length > 256:
        raise ValueError("k must be in range [1, 256]")
    num_threads = _resolve_num_threads(num_threads)
    with TemporaryDirectory() as temporary_directory:
        _call_kmc(
            [
                "kmc",
                f"-t{num_threads}",
                f"-k{kmer_length}",
                f"-ci{MINIMUM_COUNT}",
                f"-cs{CLAMP_COUNT}",
                "-r",  # RAM only mode
                read,
                output_db,
                temporary_directory,
            ],
            f"Failed to count kmers in '{read}'",
        )


def _intersect_databases(input_db1: str, input_db2: str, output_db: str):
    _call_kmc(
        ["kmc_tools", "simple", input_db1, input_db2, "intersect", output_db],
        f"Failed to intersect databases '{input_db1}' and '{input_db2}'.",
    )


def _filter_database(input_db: str, output_db: str, min_count: int):
    _call_kmc(
        ["kmc_tools", "transform", input_db, f"-ci{min_count}", "reduce", output_db],
        f"Failed to filter database '{input_db}'.",
    )


def count_kmers_single_read(
    read: str,
    output_db: str,
    kmer_length: int = 21,
    threshold: typing.Optional[float] = None,
    num_threads: typing.Optional[int] = None,
):
    if threshold is None:
        _count_kmers(read, output_db, kmer_length=kmer_length, num_threads=num_threads)
        return
    if threshold < 0.0:
        raise ValueError("threshold must be positive")
    with TemporaryDirectory() as temporary_directory:
        db1 = os.path.join(temporary_directory, "1")
        _count_kmers(read, db1, kmer_length=kmer_length, num_threads=num_threads)
        coverage = estimate_coverage(db1)
        min_count = math.ceil(threshold * coverage)
        _filter_database(db1, output_db, min_count)


def count_kmers_paired_reads(
    read1: str,
    read2: str,
    output_db: str,
    kmer_length: int = 21,
    threshold: typing.Optional[float] = None,
    num_threads: typing.Optional[int] = None,
):
    if threshold is not None and threshold < 0.0:
        raise ValueError("threshold must be positive")
    with TemporaryDirectory() as temporary_directory:
        db1 = os.path.join(temporary_directory, "1")
        _count_kmers(read1, db1, kmer_length=kmer_length, num_threads=num_threads)
        db2 = os.path.join(temporary_directory, "2")
        _count_kmers(read2, db2, kmer_length=kmer_length, num_threads=num_threads)
        if threshold is None:
            _intersect_databases(db1, db2, output_db)
            return
        db_intersection = os.path.join(temporary_directory, "intersection")
        _intersect_databases(db1, db2, db_intersection)
        coverage = estimate_coverage(db_intersection)
        min_count = math.ceil(threshold * coverage)
        _filter_database(db_intersection, output_db, min_count)
