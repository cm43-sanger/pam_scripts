import shutil

for executable in ("kmc", "kmc_tools"):
    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"Required executable '{executable}' not found in PATH."
        )

from . import _kmc

import argparse
import math
import numpy as np
import os
import subprocess
import typing
from collections.abc import Sequence
from tempfile import TemporaryDirectory

NUM_CPUS = os.cpu_count() or 1
DEFAULT_KMER_LENGTH = 21
MINIMUM_KMER_LENGTH = 1
MAXIMUM_KMER_LENGTH = 31
MINIMUM_MAX_MEMORY = 2.0
MAXIMUM_MAX_MEMORY = 1024.0
MAXIMUM_NUM_THREADS = 128
MINIMUM_COUNT = 2
CLAMP_COUNT = 65_535  # 16 bit unsigned integer maximum


def _load_kmers(db: str) -> np.ndarray[tuple[int], np.dtype[np.uint64]]:
    return _kmc.load_kmers(db)


def _estimate_coverage(db: str) -> float:
    return _kmc.estimate_coverage(db)


def _intersect_databases(input_db1: str, input_db2: str, output_db: str):
    try:
        subprocess.run(
            [
                "kmc_tools",
                "-hp",  # hide progress
                "simple",
                input_db1,
                input_db2,
                "intersect",
                output_db,
            ],
            check=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to intersect databases '{input_db1}' and '{input_db2}'."
        ) from e


def _filter_database(input_db: str, output_db: str, min_count: int):
    try:
        subprocess.run(
            [
                "kmc_tools",
                "-hp",  # hide progress
                "transform",
                input_db,
                f"-ci{min_count}",
                "reduce",
                output_db,
            ],
            check=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to filter database '{input_db}'.") from e


class KMCHelper:
    _kmer_length: int
    _threshold: float
    _max_memory: float
    _num_threads: int

    def __init__(
        self,
        kmer_length: int = DEFAULT_KMER_LENGTH,
        threshold: float = 0.0,
        max_memory: typing.Optional[float] = None,
        num_threads: typing.Optional[int] = None,
    ):
        self.kmer_length = kmer_length
        self.threshold = threshold
        self.max_memory = max_memory
        self.num_threads = num_threads

    @property
    def kmer_length(self):
        return self._kmer_length

    @kmer_length.setter
    def kmer_length(self, value: int):
        value = int(value)
        if value % 2 == 0 or value < MINIMUM_KMER_LENGTH or value > MAXIMUM_KMER_LENGTH:
            raise ValueError(
                "kmer_length must be odd and in range "
                f"[{MINIMUM_KMER_LENGTH}, {MAXIMUM_KMER_LENGTH}]"
            )
        self._kmer_length = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        value = float(value)
        if value < 0.0:
            raise ValueError("threshold must be positive")
        self._threshold = value

    @property
    def max_memory(self):
        return self._max_memory

    @max_memory.setter
    def max_memory(self, value: typing.Optional[float]):
        value = MINIMUM_MAX_MEMORY if value is None else float(value)
        if value < MINIMUM_MAX_MEMORY:
            raise ValueError(f"max_memory must be at least {MINIMUM_MAX_MEMORY} GB")
        self._max_memory = min(value, MAXIMUM_MAX_MEMORY)

    @property
    def num_threads(self):
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value: typing.Optional[int]):
        value = NUM_CPUS if value is None else int(value)
        if value < 1:
            raise ValueError("num_threads must be positive")
        self._num_threads = min(value, 128)

    def _count_kmers(self, read: str, output_db: str):
        if not os.path.exists(read):
            raise FileNotFoundError(read)
        with (
            TemporaryDirectory() as temporary_directory,
            open(f"{output_db}.log", "wb") as log_file,
        ):
            try:
                subprocess.run(
                    [
                        "kmc",
                        "-hp",  # hide progress
                        f"-t{self.num_threads}",
                        f"-k{self.kmer_length}",
                        f"-m{self.max_memory}",
                        f"-ci{MINIMUM_COUNT}",
                        f"-cs{CLAMP_COUNT}",
                        read,
                        output_db,
                        temporary_directory,
                    ],
                    stdout=log_file,
                    check=True,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to count kmers in '{read}'") from e

    def count_kmers_single_read(self, read: str, output_db: str):
        if self.threshold == 0.0:
            self._count_kmers(read, output_db)
            return
        with TemporaryDirectory() as temporary_directory:
            db1 = os.path.join(temporary_directory, "1")
            self._count_kmers(read, db1)
            coverage = _estimate_coverage(db1)
            min_count = math.ceil(self.threshold * coverage)
            _filter_database(db1, output_db, min_count)

    def count_kmers_paired_reads(self, read1: str, read2: str, output_db: str):
        with TemporaryDirectory() as temporary_directory:
            db1 = os.path.join(temporary_directory, "1")
            self._count_kmers(read1, db1)
            db2 = os.path.join(temporary_directory, "2")
            self._count_kmers(read2, db2)
            if self.threshold == 0.0:
                _intersect_databases(db1, db2, output_db)
                return
            db_intersection = os.path.join(temporary_directory, "intersection")
            _intersect_databases(db1, db2, db_intersection)
            coverage = _estimate_coverage(db_intersection)
            min_count = math.ceil(self.threshold * coverage)
            _filter_database(db_intersection, output_db, min_count)

    def count_kmers(
        self, *, read1: str, read2: typing.Optional[str] = None, output_db: str
    ):
        if read2 is None:
            self.count_kmers_single_read(read1, output_db)
        else:
            self.count_kmers_paired_reads(read1, read2, output_db)


def main():
    parser = argparse.ArgumentParser(
        description="Count kmers from one or two FASTQ/FASTA files using KMC."
    )
    parser.add_argument("-1", "--read1", required=True, help="First readset path")
    parser.add_argument("-2", "--read2", help="Second readset path (optional)")
    parser.add_argument(
        "-o", "--output_db", required=True, help="Output KMC database path"
    )
    parser.add_argument(
        "-k",
        "--kmer_length",
        type=int,
        default=DEFAULT_KMER_LENGTH,
        help=f"Kmer length (default {DEFAULT_KMER_LENGTH}, odd, "
        f">={MINIMUM_KMER_LENGTH}, <={MAXIMUM_KMER_LENGTH})",
    )
    parser.add_argument(
        "-f",
        "--threshold",
        type=float,
        default=0.0,
        help="Filter kmers with counts below threshold * coverage (default 0)",
    )
    parser.add_argument(
        "-m",
        "--max_memory",
        type=float,
        default=MINIMUM_MAX_MEMORY,
        help=f"Max amount of RAM in GB (default {MINIMUM_MAX_MEMORY}, "
        f">={MINIMUM_MAX_MEMORY})",
    )
    parser.add_argument(
        "-t",
        "--num_threads",
        type=int,
        default=NUM_CPUS,
        help=f"Number of threads (default {NUM_CPUS})",
    )
    args = parser.parse_args()

    helper = KMCHelper(
        kmer_length=args.kmer_length,
        threshold=args.threshold,
        max_memory=args.max_memory,
        num_threads=args.num_threads,
    )
    helper.count_kmers(read1=args.read1, read2=args.read2, output_db=args.output_db)


if __name__ == "__main__":
    main()
