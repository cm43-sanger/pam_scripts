import shutil

if shutil.which("kmc") is None:
    raise FileNotFoundError("Required executable 'kmc' not found in PATH.")

import numpy as np
import os
import subprocess
import typing
from collections.abc import Iterable, Sequence
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tempfile import NamedTemporaryFile, TemporaryDirectory

MINIMUM_COUNT = 2
CLAMP_COUNT = 65_535  # 16 bit unsigned integer maximum
KMER_COMPRESSOR = str.maketrans("ACGT", "0123")
KMER_DECOMPRESSOR = str.maketrans("0123", "ACGT")
MAX_THREADS = 128
NUM_CPUS = os.cpu_count() or 1


def resolve_num_threads(num_threads: typing.Optional[int]):
    if num_threads is None:
        num_threads = NUM_CPUS
    if num_threads < 0:
        raise ValueError("number of threads must be positive")
    return min(num_threads, MAX_THREADS)


def run_kmc(
    command: str, operations: Iterable[str], num_threads: typing.Optional[int] = None
):
    num_threads = resolve_num_threads(num_threads)
    subprocess_args = [command, f"-t{num_threads}"]
    subprocess_args.extend(operations)
    result = subprocess.run(subprocess_args, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"\n{subprocess_args}"
            f"\nfailed with exit code {result.returncode}. stderr:"
            f"\n{result.stderr.decode().strip()}"
        )
    return result


def count_kmers(
    reads: Sequence[str],
    counts_name: str,
    kmer_length: int = 21,
    num_threads: typing.Optional[int] = None,
):
    with (
        NamedTemporaryFile() as input_file,
        TemporaryDirectory() as temporary_directory,
    ):
        with open(input_file.name, "w") as f:
            for read in reads:
                if not os.path.exists(read):
                    raise FileNotFoundError(read)
                print(read, file=f)
        run_kmc(
            "kmc",
            [
                f"-k{kmer_length}",
                f"-ci{MINIMUM_COUNT}",
                f"-cs{CLAMP_COUNT}",
                "-r",  # RAM only mode
                f"@{input_file.name}",
                counts_name,
                temporary_directory,
            ],
            num_threads=num_threads,
        )


def get_histogram(counts_name: str, num_threads: typing.Optional[int] = None) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.uint16]],
    np.ndarray[tuple[int], np.dtype[np.uint64]],
]:
    with NamedTemporaryFile() as histogram_file:
        run_kmc(
            "kmc_tools",
            ["transform", counts_name, "-ci1", "histogram", histogram_file.name],
            num_threads=num_threads,
        )
        counts, frequencies = np.loadtxt(
            histogram_file.name,
            dtype=np.dtype([("counts", np.uint16), ("frequencies", np.uint64)]),
            delimiter="\t",
            unpack=True,
        )
    return (counts, frequencies)


class ThresholdResult(typing.NamedTuple):
    threshold: int
    counts: np.ndarray[tuple[int], np.dtype[np.uint16]]
    frequencies: np.ndarray[tuple[int], np.dtype[np.uint64]]


def threshold_histogram(counts, frequencies, sigma: float = 3.0):
    smoothed_frequencies = gaussian_filter1d(frequencies, sigma)
    peaks = find_peaks(-smoothed_frequencies, distance=sigma)
    indices = peaks[0]
    if indices.size == 0:  # failed to identify background
        return None
    min_index = indices[0]
    return ThresholdResult(
        threshold=counts[min_index],
        counts=counts[min_index:],
        frequencies=frequencies[min_index:],
    )


def summarize_histogram(counts, frequencies) -> tuple[float, float]:
    counts = np.array(counts, dtype=np.float64)
    frequencies = np.array(frequencies, dtype=np.float64)
    weights = frequencies / frequencies.sum()
    mean = np.average(counts, weights=weights)
    variance = np.average(np.square(counts - mean), weights=weights)
    return (mean, variance)


def compress_kmers(kmers: Iterable[str]):
    return (int(kmer.translate(KMER_COMPRESSOR), base=4) for kmer in kmers)


def decompress_kmers(kmers: Iterable[int], kmer_length: int):
    return (
        np.base_repr(kmer, base=4, padding=kmer_length).translate(KMER_DECOMPRESSOR)
        for kmer in kmers
    )


def filter_kmers(
    counts_name: str,
    filtered_kmers_name: str,
    threshold: int,
    num_threads: typing.Optional[int] = None,
):
    run_kmc(
        "kmc_tools",
        ["transform", counts_name, f"-ci{threshold}", "compact", filtered_kmers_name],
        num_threads=num_threads,
    )


def load_kmers(filename: str, num_threads: typing.Optional[int] = None):
    with NamedTemporaryFile() as kmer_file:
        run_kmc(
            "kmc_tools",
            ["transform", filename, "-ci1", "dump", kmer_file.name],
            num_threads=num_threads,
        )
        with open(kmer_file.name) as f:
            kmers = (line.strip().split("\t", maxsplit=1)[0] for line in f)
            return np.fromiter(compress_kmers(kmers), np.uint64)


class SketchResult(typing.NamedTuple):
    success: bool
    total: int
    threshold: typing.Optional[int] = None
    signal: typing.Optional[int] = None
    coverage: typing.Optional[float] = None
    variance: typing.Optional[float] = None


def sketch(
    reads: Sequence[str],
    filename: str,
    kmer_length: int = 21,
    num_threads: typing.Optional[int] = None,
):
    with TemporaryDirectory() as temporary_directory:
        counts_name = os.path.join(temporary_directory, "counts")
        count_kmers(
            reads, counts_name, kmer_length=kmer_length, num_threads=num_threads
        )
        counts, frequencies = get_histogram(counts_name, num_threads=num_threads)
        total = frequencies.sum()
        threshold_result = threshold_histogram(counts, frequencies)
        if threshold_result is None:
            return SketchResult(success=False, total=total)
        filter_kmers(
            counts_name, filename, threshold_result.threshold, num_threads=num_threads
        )
    coverage, variance = summarize_histogram(
        threshold_result.counts, threshold_result.frequencies
    )
    return SketchResult(
        success=True,
        total=total,
        threshold=threshold_result.threshold,
        signal=threshold_result.frequencies.sum(),
        coverage=coverage,
        variance=variance,
    )
