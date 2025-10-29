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
import warnings
import numpy as np
from dataclasses import dataclass
from tempfile import NamedTemporaryFile, TemporaryDirectory
from scipy.signal import find_peaks
from . import _kmc

NUM_CPUS = os.cpu_count() or 1
DEFAULT_KMER_LENGTH = 25
MINIMUM_KMER_LENGTH = 1
MAXIMUM_KMER_LENGTH = 31
MINIMUM_MAX_MEMORY = 2.0
MAXIMUM_MAX_MEMORY = 1024.0
MAXIMUM_NUM_THREADS = 128
MINIMUM_COUNT = 2
CLAMP_COUNT = 65_535  # 16 bit unsigned integer maximum
HIDE_PROGRESS_FLAG = "-hp"
ERROR_CORRECTION_PERCENTILE = 99.0
ERROR_CORRECTION_NUM_POINTS = 101
ERROR_CORRECTION_SHIFT = 10
ERROR_CORRECTION_WIDTH = 3


class DirectConstructionError(UserWarning):
    pass


warnings.simplefilter("error", DirectConstructionError)


def _estimate_min_count(
    counts: np.ndarray[tuple[int], np.dtype[np.uint16]],
    frequencies: np.ndarray[tuple[int], np.dtype[np.uint64]],
):
    if counts.size < 2:
        return None
    cumulative_frequencies = np.cumsum(frequencies)
    total = cumulative_frequencies[-1]
    cutoff = np.searchsorted(
        cumulative_frequencies,
        total * (ERROR_CORRECTION_PERCENTILE / 100.0),
    )
    if cutoff < 2:
        return None
    counts = counts[:cutoff]
    frequencies = frequencies[:cutoff]
    shifted_frequencies = frequencies + ERROR_CORRECTION_SHIFT
    log_counts = np.log(counts)
    log_shifted_frequencies = np.log(shifted_frequencies)
    uniform_log_counts = np.linspace(
        log_counts[0], log_counts[-1], ERROR_CORRECTION_NUM_POINTS
    )
    uniform_log_shifted_frequencies = np.interp(
        uniform_log_counts, log_counts, log_shifted_frequencies
    )
    peaks = find_peaks(
        uniform_log_shifted_frequencies,
        distance=ERROR_CORRECTION_WIDTH,
        width=ERROR_CORRECTION_WIDTH,
    )[0]
    troughs = find_peaks(
        -uniform_log_shifted_frequencies,
        distance=ERROR_CORRECTION_WIDTH,
        width=ERROR_CORRECTION_WIDTH,
    )[0]
    if troughs.size == 0 or (peaks.size != 0 and troughs[0] > peaks[0]):
        return None
    return math.exp(uniform_log_counts[troughs[0]])


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

    def correct_errors(self, output_db_path: str):
        counts, frequencies = self.histogram()
        min_count = _estimate_min_count(counts, frequencies)
        if min_count is None:
            return self.copy(output_db_path)
        return self.filter(output_db_path, min_count)


def load_database(db_path: str):
    with warnings.catch_warnings(action="ignore", category=DirectConstructionError):
        return KMCDatabase(db_path, *_kmc.get_info(db_path))


class KMCHelper:
    _kmer_length: int
    _correct_errors: bool
    _max_memory: float
    _num_threads: int

    def __init__(
        self,
        kmer_length: int = DEFAULT_KMER_LENGTH,
        correct_errors: bool = True,
        max_memory: typing.Optional[float] = None,
        num_threads: typing.Optional[int] = None,
    ):
        self.kmer_length = kmer_length
        self.correct_errors = correct_errors
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
    def correct_errors(self):
        return self._correct_errors

    @correct_errors.setter
    def correct_errors(self, value: bool):
        self._correct_errors = bool(value)

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
                    check=True,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to count kmers in {reads}") from e
        return load_database(output_db_path)

    def count_kmers(self, output_db_path: str, *reads: str):
        if not self.correct_errors:
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
        "-r", "--raw_counts", action="store_true", help=f"Don't correct kmer counts"
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
        correct_errors=not args.raw_counts,
        max_memory=args.max_memory,
        num_threads=args.num_threads,
    )
    helper.count_kmers(args.output_db_path, *args.reads)


if __name__ == "__main__":
    main()
