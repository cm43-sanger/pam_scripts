import argparse
import numpy as np
import os
import pandas as pd
import subprocess
import sys
import warnings
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import nbinom as negative_binomial
from tempfile import NamedTemporaryFile, TemporaryDirectory
from . import pam_io

MINIMUM_COUNT = 2
MAXIMUM_COUNT = 65535  # 16 bit maximum
KMER_TRANSLATOR = str.maketrans("ACGT", "0123")


def read_manifest(filename: str):
    with pam_io.get_input_handle(filename) as f:
        df = pd.read_csv(
            f, sep="\t", header=None, names=("name", "reads1", "reads2"), dtype=str
        )
    if not df["name"].is_unique:
        raise ValueError("Manifest file contains duplicate names.")
    return df


def count_kmers(
    directory: str, name: str, reads1: str, reads2: str, kmer_length: int = 21
):
    if not os.path.exists(reads1):
        raise FileNotFoundError(reads1)
    if not os.path.exists(reads2):
        raise FileNotFoundError(reads2)
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
                f"@{input_file.name}",
                counts_name,
                directory,
            ],
            capture_output=True,
        )
    if result.returncode:
        raise RuntimeError("Failed to count kmers with kmc")
    return counts_name


def get_histogram(counts_name: str):
    with NamedTemporaryFile() as histogram_file:
        result = subprocess.run(
            ["kmc_tools", "transform", counts_name, "histogram", histogram_file.name],
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError("Failed to get histogram from kmc database")
        return np.loadtxt(histogram_file.name, dtype=np.uint64, delimiter="\t").T


def get_threshold(counts, frequencies, right: float = 0.9, sigma: float = 0.01):
    cumulative_frequencies = np.cumsum(frequencies)
    total = cumulative_frequencies[-1]
    stop = np.searchsorted(cumulative_frequencies, right * total)
    counts = counts[:stop]
    frequencies = frequencies[:stop]
    sigma = sigma * counts.max()
    smoothed_frequencies = gaussian_filter1d(frequencies, sigma)
    peaks = find_peaks(-smoothed_frequencies, distance=sigma)
    indices = peaks[0]
    if indices.size == 0:
        raise ValueError("Failed to distinguish between signal and background kmers")
    min_index = indices[0]
    threshold = counts[min_index]
    ratio = (total - frequencies[min_index - 1]) / total
    counts = counts[min_index:]
    frequencies = frequencies[min_index:]
    coverage = np.average(counts, weights=frequencies)
    return (threshold, ratio, coverage)


def filter_kmers(counts_name: str, threshold: int):
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
            return np.fromiter(
                (
                    int(
                        line.strip()
                        .split("\t", maxsplit=1)[0]
                        .translate(KMER_TRANSLATOR),
                        4,
                    )
                    for line in f
                ),
                np.uint64,
            )


def sketch(name: str, reads1: str, reads2: str, kmer_length: int = 21):
    with TemporaryDirectory() as temporary_directory:
        counts_name = count_kmers(
            temporary_directory, name, reads1, reads2, kmer_length
        )
        counts, frequencies = get_histogram(counts_name)
        threshold, ratio, coverage = get_threshold(counts, frequencies)
        kmers = filter_kmers(counts_name, threshold)
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
