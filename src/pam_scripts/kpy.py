import argparse
import numpy as np
import os
import pandas as pd
import subprocess
import typing
import warnings
from collections.abc import Iterable
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tempfile import NamedTemporaryFile, TemporaryDirectory
from . import pam_io

MINIMUM_COUNT = 2
MAXIMUM_COUNT = 65535  # 16 bit maximum
KMER_COMPRESSOR = str.maketrans("ACGT", "0123")
KMER_DECOMPRESSOR = str.maketrans("0123", "ACGT")
NUM_CPUS = os.cpu_count() or 1


def resolve_num_threads(num_threads: typing.Optional[int]):
    if num_threads is None:
        return NUM_CPUS
    if num_threads < 0:
        raise ValueError("number of threads must be positive")
    return num_threads


def read_manifest(filename: str):
    with pam_io.get_input_handle(filename) as f:
        df = pd.read_csv(
            f, sep="\t", header=None, names=("name", "reads1", "reads2"), dtype=str
        )
    if not df["name"].is_unique:
        raise ValueError("Manifest file contains duplicate names.")
    return df


def count_kmers(
    directory: str,
    name: str,
    reads1: str,
    reads2: str,
    kmer_length: int = 21,
    num_threads: typing.Optional[int] = None,
):
    if not os.path.exists(reads1):
        raise FileNotFoundError(reads1)
    if not os.path.exists(reads2):
        raise FileNotFoundError(reads2)
    num_threads = resolve_num_threads(num_threads)
    if num_threads > 128:
        warnings.warn("Reducing number of kmer counting threads to 128")
        num_threads = 128
    counts_name = os.path.join(directory, name)
    with NamedTemporaryFile() as input_file:
        with open(input_file.name, "w") as f:
            f.write(f"{reads1}\n{reads2}\n")
        result = subprocess.run(
            [
                "kmc",
                f"-k{kmer_length}",
                f"-cs{MAXIMUM_COUNT}",
                "-r",  # RAM only mode
                f"-t{num_threads}",
                f"@{input_file.name}",
                counts_name,
                directory,
            ],
            capture_output=True,
        )
    if result.returncode:
        raise RuntimeError("Failed to count kmers with kmc")
    return counts_name


def get_histogram(counts_name: str, num_threads: typing.Optional[int] = None):
    num_threads = resolve_num_threads(num_threads)
    with NamedTemporaryFile() as histogram_file:
        result = subprocess.run(
            [
                "kmc_tools",
                f"-t{num_threads}",
                "transform",
                counts_name,
                "histogram",
                histogram_file.name,
            ],
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError("Failed to get histogram from kmc database")
        return np.loadtxt(histogram_file.name, dtype=np.uint64, delimiter="\t").T


def get_threshold(counts, frequencies, sigma: float = 3.0):
    smoothed_frequencies = gaussian_filter1d(frequencies, sigma)
    peaks = find_peaks(-smoothed_frequencies, distance=sigma)
    indices = peaks[0]
    if indices.size == 0:
        # all signal
        # this is extremely unlikely but resolve downstream
        threshold = MINIMUM_COUNT
        ratio = 1.0
    else:
        # factor out background
        min_index = indices[0]
        threshold = counts[min_index]
        num_background = frequencies[:min_index].sum()
        num_signal = frequencies[min_index:].sum()
        ratio = num_signal / (num_background + num_signal)
        counts = counts[min_index:]
        frequencies = frequencies[min_index:]
    coverage = np.average(counts, weights=frequencies)
    return (threshold, ratio, coverage)


def compress_kmers(kmers: Iterable[str]):
    return (int(kmer.translate(KMER_COMPRESSOR), base=4) for kmer in kmers)


def decompress_kmers(kmers: Iterable[int], kmer_length: int):
    return (
        np.base_repr(kmer, base=4, padding=kmer_length).translate(KMER_DECOMPRESSOR)
        for kmer in kmers
    )


def filter_kmers(
    counts_name: str, threshold: int, num_threads: typing.Optional[int] = None
):
    num_threads = resolve_num_threads(num_threads)
    with NamedTemporaryFile() as kmer_file:
        result = subprocess.run(
            [
                "kmc_tools",
                "transform",
                counts_name,
                f"-ci{threshold}",
                "dump",
                kmer_file.name,
            ],
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError("Failed to filter kmers")
        with open(kmer_file.name) as f:
            kmers = (line.strip().split("\t", maxsplit=1)[0] for line in f)
            return np.fromiter(compress_kmers(kmers), np.uint64)


def sketch(
    name: str,
    reads1: str,
    reads2: str,
    kmer_length: int = 21,
    num_threads: typing.Optional[int] = None,
):
    with TemporaryDirectory() as temporary_directory:
        counts_name = count_kmers(
            temporary_directory,
            name,
            reads1,
            reads2,
            kmer_length=kmer_length,
            num_threads=num_threads,
        )
        counts, frequencies = get_histogram(counts_name, num_threads=num_threads)
        threshold, ratio, coverage = get_threshold(counts, frequencies)
        kmers = filter_kmers(counts_name, threshold, num_threads=num_threads)
    return (counts, frequencies, threshold, ratio, coverage, kmers)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "manifest",
        nargs="?",
        default="-",
        help="Input manifest file (defaults to stdin)",
    )
    args = parser.parse_args()

    df = read_manifest(args.manifest)

    for row in df.itertuples(index=False):
        print(row.name, row.reads1, row.reads2, sep="\t")


if __name__ == "__main__":
    main()
