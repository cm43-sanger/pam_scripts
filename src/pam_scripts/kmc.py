from __future__ import annotations

import shutil

for executable in ("kmc", "kmc_tools"):
    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"Required executable '{executable}' not found in PATH."
        )

import argparse
import math
import os
import subprocess
import typing
import numpy as np
from tempfile import NamedTemporaryFile, TemporaryDirectory
from scipy.signal import find_peaks
from . import _core, _kmc

NUM_CPUS = os.cpu_count() or 1
DEFAULT_KMER_LENGTH = 25
MINIMUM_KMER_LENGTH = 1
MAXIMUM_KMER_LENGTH = 31
DEFAULT_THRESHOLD = 0.05
MINIMUM_MAX_MEMORY = 2.0
MAXIMUM_MAX_MEMORY = 1024.0
MAXIMUM_NUM_THREADS = 128
MINIMUM_COUNT = 2
CLAMP_COUNT = 65_535  # 16 bit unsigned integer maximum
HIDE_PROGRESS_FLAG = "-hp"


@_core.guarded_dataclass
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

    def load_kmers(self) -> np.ndarray[tuple[int], np.dtype[np.uint64]]:
        return _kmc.load_kmers(self.path)

    def histogram(
        self,
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.uint16]],
        np.ndarray[tuple[int], np.dtype[np.uint64]],
    ]:
        with NamedTemporaryFile() as histogram_file:
            try:
                subprocess.run(
                    [
                        "kmc_tools",
                        HIDE_PROGRESS_FLAG,
                        "transform",
                        self.path,
                        "histogram",
                        histogram_file.name,
                    ],
                    check=True,
                )
                counts, frequencies = np.loadtxt(
                    histogram_file.name,
                    dtype=np.dtype([("counts", np.uint16), ("frequencies", np.uint64)]),
                    delimiter="\t",
                    unpack=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate histogram for database {self.path!r}."
                ) from e
        return (counts, frequencies)

    def estimate_coverage(self):
        counts, frequencies = self.histogram()
        if counts.size < 2:
            raise ValueError("not enough kmers to estimate coverage")
        cumulative_frequencies = np.cumsum(frequencies)
        total_kmers = cumulative_frequencies[-1]
        cutoff99 = np.searchsorted(cumulative_frequencies, 0.99 * total_kmers)
        low_cutoff = np.searchsorted(counts, math.sqrt(counts[cutoff99]))
        return float(
            np.average(
                counts[low_cutoff:cutoff99], weights=frequencies[low_cutoff:cutoff99]
            )
        )

    def copy(self, output_db_path: str):
        shutil.copyfile(f"{self.path}.kmc_pre", f"{output_db_path}.kmc_pre")
        shutil.copyfile(f"{self.path}.kmc_suf", f"{output_db_path}.kmc_suf")
        return load_database(output_db_path)

    def filter(self, output_db_path: str, min_count: float):
        if min_count < self.min_count:
            raise ValueError(
                f"provided min_count ({min_count}) less than database min_count "
                f"({self.min_count})"
            )
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
            raise RuntimeError(f"Failed to filter database {self.path!r}.") from e
        return load_database(output_db_path)

    def correct_errors(self, output_db_path: str, threshold: float = DEFAULT_THRESHOLD):
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        if threshold == 0.0:
            return self.copy(output_db_path)
        coverage = self.estimate_coverage()
        min_count = coverage * threshold
        if min_count < self.min_count:
            return self.copy(output_db_path)
        return self.filter(output_db_path, min_count)


def load_database(db_path: str):
    with _core.release_guard():
        return KMCDatabase(db_path, *_kmc.get_info(db_path))


class KMCHelper:
    _kmer_length: int
    _threshold: float
    _max_memory: float
    _num_threads: int

    def __init__(
        self,
        kmer_length: int = DEFAULT_KMER_LENGTH,
        threshold: float = DEFAULT_THRESHOLD,
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
            if (
                value % 2 == 0
                or value < MINIMUM_KMER_LENGTH
                or value > MAXIMUM_KMER_LENGTH
            ):
                raise ValueError
        except ValueError:
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
            if value < 0.0:
                raise ValueError
        except:
            raise ValueError(f"threshold must be positive (got {value})")
        self._threshold = round(value, 6)

    @property
    def max_memory(self):
        return self._max_memory

    @max_memory.setter
    def max_memory(self, value: typing.Optional[float]):
        try:
            value = MINIMUM_MAX_MEMORY if value is None else float(value)
            if value < MINIMUM_MAX_MEMORY:
                raise ValueError
        except ValueError:
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
            if value <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"num_threads must be a positive integer (got {value})")
        self._num_threads = min(value, 128)

    def _count_raw_kmers(self, output_db_path: str, *reads: str):
        if not reads:
            raise ValueError("At least one read file must be provided")
        for read in reads:
            if not os.path.exists(read):
                raise FileNotFoundError(read)
        with (
            TemporaryDirectory() as temporary_directory,
            NamedTemporaryFile() as manifest_file,
            open(f"{output_db_path}.log", "wb") as log_file,
        ):
            with open(manifest_file.name, "w") as f:
                for read in reads:
                    print(read, file=f)
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
                        f"@{manifest_file.name}",
                        output_db_path,
                        temporary_directory,
                    ],
                    stdout=log_file,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"\n{e.stderr}Failed to count kmers in {reads}"
                ) from e
            except Exception as e:
                raise RuntimeError(f"failed to count kmers in {reads}") from e
        return load_database(output_db_path)

    def count_kmers(self, output_db_path: str, *reads: str):
        if self.threshold == 0.0:
            return self._count_raw_kmers(output_db_path, *reads)
        with TemporaryDirectory() as temporary_directory:
            raw_db_path = os.path.join(temporary_directory, "raw_counts")
            raw_db = self._count_raw_kmers(raw_db_path, *reads)
            return raw_db.correct_errors(output_db_path)


def main():
    parser = argparse.ArgumentParser(
        description="Count kmers from one or two FASTQ/FASTA files using KMC."
    )
    parser.add_argument("output_db_path", help="Output KMC database path")
    parser.add_argument("reads", nargs="+", help="Readset path(s)")
    parser.add_argument(
        "-k",
        "--kmer_length",
        type=int,
        default=DEFAULT_KMER_LENGTH,
        help=f"Kmer length (default {DEFAULT_KMER_LENGTH}, odd, "
        f">={MINIMUM_KMER_LENGTH}, <={MAXIMUM_KMER_LENGTH})",
    )
    parser.add_argument(
        "-c",
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Filter kmers with counts below threshold * coverage (default "
        f"{DEFAULT_THRESHOLD})",
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
    helper.count_kmers(args.output_db_path, *args.reads)


if __name__ == "__main__":
    main()
