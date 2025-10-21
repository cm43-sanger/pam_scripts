from __future__ import annotations

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
import warnings
from dataclasses import dataclass
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
HIDE_PROGRESS_FLAG = "-hp"


class DirectConstructionError(UserWarning):
    pass


warnings.simplefilter("error", DirectConstructionError)


@dataclass(frozen=True)
class KMCDatabase:
    path: str
    kmer_length: int
    mode: int
    counter_size: int
    lut_prefix_length: int
    signature_len: int
    min_count: int
    max_count: int
    both_strands: bool
    total_kmers: int

    def __post_init__(self):
        warnings.warn("construct with load_database", DirectConstructionError)

    def load_kmers(self) -> np.ndarray[tuple[int], np.dtype[np.uint64]]:
        return _kmc.load_kmers(self.path)

    def estimate_coverage(self) -> float:
        return _kmc.estimate_coverage(self.path)

    def filter(self, min_count: float, output_db_path: str):
        if min_count < 0.0:
            raise ValueError(f"min_count must be non-negative, got {min_count}")
        try:
            subprocess.run(
                [
                    "kmc_tools",
                    HIDE_PROGRESS_FLAG,
                    "transform",
                    self.path,
                    f"-ci{math.ceil(min_count)}",
                    "reduce",
                    output_db_path,
                ],
                check=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to filter database '{self.path}'.") from e
        return load_database(output_db_path)

    def intersect(self, other_db: KMCDatabase, output_db_path: str):
        if self.kmer_length != other_db.kmer_length:
            raise ValueError(
                f"KMC databases have mismatched kmer_length: "
                f"{self.kmer_length} != {other_db.kmer_length}"
            )
        try:
            subprocess.run(
                [
                    "kmc_tools",
                    HIDE_PROGRESS_FLAG,
                    "simple",
                    self.path,
                    other_db.path,
                    "intersect",
                    output_db_path,
                ],
                check=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to intersect databases '{self.path}' and '{other_db.path}'."
            ) from e
        return load_database(output_db_path)


def load_database(db_path: str):
    with warnings.catch_warnings(action="ignore", category=DirectConstructionError):
        return KMCDatabase(db_path, *_kmc.get_info(db_path))


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
        try:
            value = int(value)
            assert (
                value % 2
                and value >= MINIMUM_KMER_LENGTH
                and value <= MAXIMUM_KMER_LENGTH
            )
        except:
            raise ValueError(
                "kmer_length must be a positive, odd integer in range "
                f"[{MINIMUM_KMER_LENGTH}, {MAXIMUM_KMER_LENGTH}] (got {value})"
            )
        self._kmer_length = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        try:
            value = float(value)
            assert value >= 0.0
        except:
            raise ValueError(f"threshold must be positive (got {value})")
        self._threshold = value

    @property
    def max_memory(self):
        return self._max_memory

    @max_memory.setter
    def max_memory(self, value: typing.Optional[float]):
        try:
            value = MINIMUM_MAX_MEMORY if value is None else float(value)
            assert value >= MINIMUM_MAX_MEMORY
        except:
            raise ValueError(
                f"max_memory must be at least {MINIMUM_MAX_MEMORY} GB (got {value})"
            )
        self._max_memory = min(value, MAXIMUM_MAX_MEMORY)

    @property
    def num_threads(self):
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value: typing.Optional[int]):
        try:
            value = NUM_CPUS if value is None else int(value)
            assert value > 0
        except:
            raise ValueError(f"num_threads must be a positive integer (got {value})")
        self._num_threads = min(value, 128)

    def _count_kmers(self, read: str, output_db_path: str):
        if not os.path.exists(read):
            raise FileNotFoundError(read)
        with (
            TemporaryDirectory() as temporary_directory,
            open(f"{output_db_path}.log", "wb") as log_file,
        ):
            try:
                subprocess.run(
                    [
                        "kmc",
                        HIDE_PROGRESS_FLAG,
                        f"-t{self.num_threads}",
                        f"-k{self.kmer_length}",
                        f"-m{self.max_memory}",
                        f"-ci{MINIMUM_COUNT}",
                        f"-cs{CLAMP_COUNT}",
                        read,
                        output_db_path,
                        temporary_directory,
                    ],
                    stdout=log_file,
                    check=True,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to count kmers in '{read}'") from e
        return load_database(output_db_path)

    def count_kmers_single_read(self, read: str, output_db_path: str):
        if self.threshold == 0.0:
            return self._count_kmers(read, output_db_path)
        with TemporaryDirectory() as temporary_directory:
            db1 = self._count_kmers(read, os.path.join(temporary_directory, "1"))
            coverage = db1.estimate_coverage()
            return db1.filter(self.threshold * coverage, output_db_path)

    def count_kmers_paired_reads(self, read1: str, read2: str, output_db_path: str):
        with TemporaryDirectory() as temporary_directory:
            db1 = self._count_kmers(read1, os.path.join(temporary_directory, "1"))
            db2 = self._count_kmers(read2, os.path.join(temporary_directory, "2"))
            if self.threshold == 0.0:
                return db1.intersect(db2, output_db_path)
            db_intersection = db1.intersect(
                db2, os.path.join(temporary_directory, "intersection")
            )
            coverage = db_intersection.estimate_coverage()
            return db_intersection.filter(self.threshold * coverage, output_db_path)

    def count_kmers(
        self, *, read1: str, read2: typing.Optional[str] = None, output_db_path: str
    ):
        if read2 is None:
            return self.count_kmers_single_read(read1, output_db_path)
        return self.count_kmers_paired_reads(read1, read2, output_db_path)


def main():
    parser = argparse.ArgumentParser(
        description="Count kmers from one or two FASTQ/FASTA files using KMC."
    )
    parser.add_argument("-1", "--read1", required=True, help="First readset path")
    parser.add_argument("-2", "--read2", help="Second readset path (optional)")
    parser.add_argument(
        "-o", "--output_db_path", required=True, help="Output KMC database path"
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
    helper.count_kmers(
        read1=args.read1, read2=args.read2, output_db_path=args.output_db_path
    )


if __name__ == "__main__":
    main()
