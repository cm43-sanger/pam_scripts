from . import _kmc, kmers

import argparse
import multiprocessing
import numpy as np
import os
import pandas as pd
import shutil
import sys
import threading
import time
import typing
import warnings
from collections.abc import Sequence
from contextlib import redirect_stdout
from numba import njit, prange
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tempfile import NamedTemporaryFile, TemporaryDirectory
from tqdm import tqdm as make_progressbar

MINIMUM_COUNT = 2
CLAMP_COUNT = 65_535  # 16 bit unsigned integer maximum
NUM_CPUS = os.cpu_count() or 1


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
        _kmc.call_kmc(
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
    # with NamedTemporaryFile() as histogram_file:
    #     _kmc.call_kmc_tools(
    #         ["transform", counts_name, "-ci1", "histogram", histogram_file.name],
    #         num_threads=num_threads,
    #     )
    #     counts, frequencies = np.loadtxt(
    #         histogram_file.name,
    #         dtype=np.dtype([("counts", np.uint16), ("frequencies", np.uint64)]),
    #         delimiter="\t",
    #         unpack=True,
    #     )
    histogram_filename = f"{counts_name}.hist.dat"
    _kmc.call_kmc_tools(
        ["transform", counts_name, "-ci1", "histogram", histogram_filename],
        num_threads=num_threads,
    )
    counts, frequencies = np.loadtxt(
        histogram_filename,
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


def filter_counts(
    counts_name: str,
    filtered_kmers_name: str,
    threshold: int,
    num_threads: typing.Optional[int] = None,
):
    _kmc.call_kmc_tools(
        [
            "transform",
            counts_name,
            f"-ci{threshold}",
            "set_counts",
            str(CLAMP_COUNT),
            filtered_kmers_name,
        ],
        num_threads=num_threads,
    )


class SketchResult(typing.NamedTuple):
    success: bool
    total: typing.Optional[int] = -1
    signal: typing.Optional[int] = -1
    threshold: typing.Optional[int] = -1
    coverage: typing.Optional[float] = -1.0
    variance: typing.Optional[float] = -1.0


def sketch_reads(
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
        filter_counts(
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


def load_manifest(manifest: str):
    try:
        with open(manifest) as f:
            lines = [line.strip() for line in f]
        names: list[str] = []
        for i, line in enumerate(lines):
            try:
                name, _ = line.split("\t", maxsplit=1)
            except ValueError:
                raise ValueError(
                    f"Line {i} has less than two entries, needs name and read file(s)"
                )
            names.append(name)
    except Exception as e:
        raise ValueError(f"unable to load manifest '{manifest}'") from e
    return (lines, names)


__sketch_from_manifest_store: typing.Optional[tuple[str, dict[str, typing.Any]]] = None


def __sketch_from_manifest_worker_init(sketches_directory: str, kwargs: dict):
    global __sketch_from_manifest_store
    __sketch_from_manifest_store = (sketches_directory, kwargs)


def __sketch_from_manifest_worker_func(line: str):
    if __sketch_from_manifest_store is None:
        raise RuntimeError(
            "worker function called outside of initialized multiprocessing context."
        )
    name, *reads = line.split("\t")
    sketches_directory, kwargs = __sketch_from_manifest_store
    filename = os.path.join(sketches_directory, name)
    try:
        result = sketch_reads(reads, filename, **kwargs)
    except:
        result = SketchResult(success=False)
    return result


def _resolve_num_threads(
    num_jobs: typing.Optional[int] = None, num_kmc_threads: typing.Optional[int] = None
):
    warnings.warn("Use total thread budget rather than num_kmc_threads")
    if num_jobs is not None and num_jobs < 1:
        raise ValueError("the number of jobs must be positive")
    if num_kmc_threads is not None and num_kmc_threads < 1:
        raise ValueError("the number of KMC threads must be positive")
    if num_jobs is None:
        if num_kmc_threads is None:
            return (1, NUM_CPUS)
        return (NUM_CPUS // num_kmc_threads, num_kmc_threads)
    num_jobs = min(num_jobs, NUM_CPUS)
    if num_kmc_threads is None or num_jobs * num_kmc_threads > NUM_CPUS:
        return (num_jobs, NUM_CPUS // num_jobs)
    return (num_jobs, num_kmc_threads)


def sketch_from_manifest(
    manifest: str,
    directory: str,
    kmer_length: int = 21,
    num_jobs: typing.Optional[int] = None,
    num_kmc_threads: typing.Optional[int] = None,
    verbose: bool = False,
):
    num_jobs, num_kmc_threads = _resolve_num_threads(num_jobs, num_kmc_threads)
    if verbose:
        print(
            f"Sketching '{manifest}' with {num_jobs} jobs, "
            f"each with {num_kmc_threads} threads.",
            file=sys.stderr,
        )
    lines, names = load_manifest(manifest)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    sketches_directory = os.path.join(directory, "sketches")
    os.makedirs(sketches_directory)  # also creates parent directory
    with open(os.path.join(directory, "manifest.tsv"), "w") as f:
        for line in lines:
            print(line, file=f)
    kwargs = dict(kmer_length=kmer_length, num_threads=num_kmc_threads)
    num_failures = 0
    with (
        multiprocessing.Pool(
            num_jobs,
            initializer=__sketch_from_manifest_worker_init,
            initargs=(sketches_directory, kwargs),
        ) as pool,
        make_progressbar(
            desc="Sketching",
            total=len(lines),
            disable=not verbose,
            postfix={"failures": 0},
        ) as progressbar,
        open(os.path.join(directory, "results.tsv"), "w") as f,
    ):
        print("name", "\t".join(SketchResult._fields), sep="\t", file=f)
        for name, result in zip(
            names, pool.imap(__sketch_from_manifest_worker_func, lines)
        ):
            if not result.success:
                num_failures += 1
                progressbar.set_postfix({"failures": num_failures})
            print(name, "\t".join(map(str, result)), sep="\t", file=f)
            progressbar.update()
    return len(lines)


def __load_sketches_worker_func(filename: str):
    return kmers.load_kmers(filename, num_threads=1)


@njit
def _jaccard_similarity_numba(a, b):
    """Compute Jaccard similarity between two sorted uint64 arrays."""
    i = 0
    j = 0
    intersection = 0
    len_a = a.size
    len_b = b.size
    while i < len_a and j < len_b:
        ai = a[i]
        bj = b[j]
        intersection += ai == bj
        i += ai <= bj
        j += ai >= bj
    union = len_a + len_b - intersection
    if union == 0:
        return 0.0
    return intersection / union


@njit(parallel=True)
def _pairwise_jaccard_numba(n, arrays, d, progress):
    # Compute upper triangle in parallel
    for i in prange(n):
        print(i)
        d[i, i] = 1.0  # diagonal
        for j in range(i + 1, n):
            sim = _jaccard_similarity_numba(arrays[i], arrays[j])
            d[i, j] = sim
            d[j, i] = sim  # symmetric


def pairwise_jaccard(arrays):
    """
    Compute pairwise Jaccard similarity between a list of sorted uint64 arrays.
    Returns a symmetric float64 matrix.
    """
    n = len(arrays)
    d = np.empty((n, n), dtype=np.float64)
    total = n * (n - 1) // 2
    with make_progressbar(total=total) as progressbar:
        progress = np.zeros(1, dtype=np.int64)
        thread = threading.Thread(
            target=_pairwise_jaccard_numba, args=(n, arrays, d, progress)
        )
        thread.start()
        last = 0
        while thread.is_alive():
            current = progress[0]
            progressbar.update(current - last)
            last = current
            time.sleep(0.01)
        thread.join()
    return d


def load_sketches(directory: str):
    results = pd.read_csv(os.path.join(directory, "results.tsv"), sep="\t")
    unsuccessful_names = []
    names = []
    sketches_directory = os.path.join(directory, "sketches")
    for row in results.itertuples():
        if row.success:
            names.append(row.name)
        else:
            unsuccessful_names.append(row.name)
    filenames = (os.path.join(sketches_directory, name) for name in names)
    with (
        multiprocessing.Pool() as pool,
        make_progressbar(
            pool.imap(__load_sketches_worker_func, filenames),
            desc="Loading databases",
            total=len(names),
        ) as progressbar,
    ):
        kmers_list_lookup = {
            name: kmers_list for name, kmers_list in zip(names, progressbar)
        }
    results["kmers"] = results["name"].map(kmers_list_lookup)
    return results


# def calculate_distances(results: pd.DataFrame):
#     mask =
#     num = results['']


def main():
    parser = argparse.ArgumentParser(
        description="Generate kmer sketches from a manifest of read sets."
    )
    parser.add_argument(
        "manifest",
        help="Path to the manifest file listing input read sets (TSV format)",
    )
    parser.add_argument(
        "directory", help="Output directory to store the generated sketches"
    )
    parser.add_argument(
        "--kmer-length",
        "-k",
        type=int,
        default=21,
        help="K-mer length to use for sketching (default: 21)",
    )
    parser.add_argument(
        "--num-jobs",
        "-j",
        type=int,
        default=None,
        help="Number of parallel jobs (default: auto)",
    )
    parser.add_argument(
        "--num-kmc-threads",
        "-t",
        type=int,
        default=None,
        help="Threads per job for KMC (default: auto)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose progress output"
    )
    args = parser.parse_args()

    sketch_from_manifest(
        args.manifest,
        args.directory,
        kmer_length=args.kmer_length,
        num_jobs=args.num_jobs,
        num_kmc_threads=args.num_kmc_threads,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
