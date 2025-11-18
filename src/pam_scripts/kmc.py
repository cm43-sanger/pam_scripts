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
from . import _core, _kmc

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
    _homopolymer: bool
    _threshold: float
    _max_memory: float
    _num_threads: int

    def __init__(
        self,
        kmer_length: int = DEFAULT_KMER_LENGTH,
        homopolymer: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
        max_memory: typing.Optional[float] = None,
        num_threads: typing.Optional[int] = None,
    ):
        self.kmer_length = kmer_length
        self.homopolymer = homopolymer
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
    def homopolymer(self):
        return self._homopolymer

    @homopolymer.setter
    def homopolymer(self, value: bool):
        self._homopolymer = bool(value)

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
            value = _core.NUM_CPUS if value is None else int(value)
            if value <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"num_threads must be a positive integer (got {value})")
        self._num_threads = min(value, 128)

    # def _count_raw_kmers(
    #     self, output_db_path: str, input_path: str, min_count: int, format_arg: str
    # ):
    #     with (
    #         TemporaryDirectory() as temporary_directory,
    #         open(f"{output_db_path}.log", "wb") as log_file,
    #     ):
    #         try:
    #             args = [
    #                 HIDE_PROGRESS_FLAG,
    #                 f"-t{self.num_threads}",
    #                 f"-k{self.kmer_length}",
    #                 f"-m{self.max_memory}",
    #                 f"-ci{min_count}",
    #                 f"-cs{CLAMP_COUNT}",
    #                 format_arg,
    #             ]
    #             if self.homopolymer:
    #                 args.append("-hc")
    #             subprocess.run(
    #                 ["kmc"] + args + [input_path, output_db_path, temporary_directory],
    #                 stdout=log_file,
    #                 stderr=subprocess.PIPE,
    #                 check=True,
    #                 text=True,
    #             )
    #         except subprocess.CalledProcessError as e:
    #             raise RuntimeError(
    #                 f"\n{e.stderr}Failed to count kmers in {input_path}"
    #             ) from e
    #         except Exception as e:
    #             raise RuntimeError(f"failed to count kmers in {input_path}") from e
    #     return load_database(output_db_path)

    # def count_kmers_fasta(self, output_db_path: str, fasta_path: str):
    #     if not os.path.exists(fasta_path):
    #         raise FileNotFoundError(fasta_path)
    #     return self._count_raw_kmers(
    #         output_db_path, input_path=fasta_path, min_count=1, format_arg="-fa"
    #     )

    # def _count_raw_kmers_fastq(self, output_db_path: str, *fastq_paths: str):
    #     if not fastq_paths:
    #         raise ValueError("At least one fastq file must be provided")
    #     with NamedTemporaryFile() as manifest_file:
    #         with open(manifest_file.name, "w") as f:
    #             for path in fastq_paths:
    #                 if not os.path.exists(path):
    #                     raise FileNotFoundError(path)
    #                 print(path, file=f)
    #         try:
    #             return self._count_raw_kmers(
    #                 output_db_path,
    #                 input_path=f"@{manifest_file.name}",
    #                 min_count=MINIMUM_COUNT,
    #                 format_arg="-fq",
    #             )
    #         except Exception as e:
    #             raise RuntimeError(f"failed to count kmers in {fastq_paths}") from e

    # def count_kmers_fastq(self, output_db_path: str, *fastq_paths: str):
    #     if self.threshold == 0.0:
    #         return self._count_raw_kmers_fastq(output_db_path, *fastq_paths)
    #     with TemporaryDirectory() as temporary_directory:
    #         raw_db_path = os.path.join(temporary_directory, "raw_counts")
    #         raw_db = self._count_raw_kmers_fastq(raw_db_path, *fastq_paths)
    #         return raw_db.correct_errors(output_db_path)

    def _count_raw_kmers(
        self,
        output_db_path: str,
        input_paths: tuple[str, ...],
        min_count: int,
        format_arg: str,
    ):
        with (
            NamedTemporaryFile() as manifest_file,
            TemporaryDirectory() as temporary_directory,
            open(f"{output_db_path}.log", "wb") as log_file,
        ):
            with open(manifest_file.name, "w") as f:
                for path in input_paths:
                    if not os.path.exists(path):
                        raise FileNotFoundError(path)
                    print(path, file=f)
            try:
                args = [
                    HIDE_PROGRESS_FLAG,
                    f"-t{self.num_threads}",
                    f"-k{self.kmer_length}",
                    f"-m{self.max_memory}",
                    f"-ci{min_count}",
                    f"-cs{CLAMP_COUNT}",
                    format_arg,
                ]
                if self.homopolymer:
                    args.append("-hc")
                subprocess.run(
                    ["kmc"]
                    + args
                    + [f"@{manifest_file.name}", output_db_path, temporary_directory],
                    stdout=log_file,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"\n{e.stderr}Failed to count kmers in {input_paths}"
                ) from e
            except Exception as e:
                raise RuntimeError(f"failed to count kmers in {input_paths}") from e
        return load_database(output_db_path)

    def count_kmers_fasta(self, output_db_path: str, *fasta_paths: str):
        return self._count_raw_kmers(
            output_db_path, fasta_paths, min_count=1, format_arg="-fa"
        )

    def count_kmers_fastq(self, output_db_path: str, *fastq_paths: str):
        return self._count_raw_kmers(
            output_db_path, fastq_paths, min_count=MINIMUM_COUNT, format_arg="-fq"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Count kmers from one or more fasta/fastq files using KMC."
    )
    parser.add_argument("output_db_path", help="Output KMC database path")
    parser.add_argument("input_paths", nargs="+", help="Input path(s)")
    parser.add_argument(
        "-f",
        "--input_format",
        type=str,
        default="q",
        choices=("a", "q"),
        metavar="[a|q]",
        help="Input format: 'a' for fasta, 'q' for fastq (default fastq)",
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
        "-hc",
        "--homopolymer",
        action="store_true",
        help=f"Enable homopolymer compression (default False)",
    )
    parser.add_argument(
        "-c",
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Filter kmers in fastq(s) with counts below threshold * coverage "
        f"(default {DEFAULT_THRESHOLD})",
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
        default=_core.NUM_CPUS,
        help=f"Number of threads (default {_core.NUM_CPUS})",
    )
    args = parser.parse_args()

    helper = KMCHelper(
        kmer_length=args.kmer_length,
        homopolymer=args.homopolymer,
        threshold=args.threshold,
        max_memory=args.max_memory,
        num_threads=args.num_threads,
    )
    if args.input_format == "a":
        helper.count_kmers_fasta(args.output_db_path, *args.input_paths)
    else:
        helper.count_kmers_fastq(args.output_db_path, *args.input_paths)


if __name__ == "__main__":
    main()
