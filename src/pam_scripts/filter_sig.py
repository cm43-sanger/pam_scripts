import argparse
import gzip
import json
import operator
import numpy as np
from collections import Counter
from . import pam_io

CUTOFF = 0.99


def load_signatures(path: str):
    try:
        with gzip.open(path) as f:
            return json.load(f)
    except gzip.BadGzipFile:
        with open(path) as f:
            return json.load(f)


def get_histogram(signature, min_count: int = 2):
    if "abundances" not in signature:
        raise ValueError("abundance needed for histogram")
    if not signature["mins"]:
        return (np.array([]), np.array([]))
    if len(signature["mins"]) != len(signature["abundances"]):
        raise ValueError("length mismatch between mins and abundances")
    counter = Counter(signature["abundances"])
    for count in range(1, min_count):
        counter.pop(count, 0)
    if not counter:
        return (np.array([]), np.array([]))
    count, frequency = zip(*sorted(counter.items(), key=operator.itemgetter(0)))
    return (np.array(count), np.array(frequency))


def estimate_coverage(signature, min_count: int = 2, cutoff: float = CUTOFF):
    count, frequency = get_histogram(signature, min_count=min_count)
    if len(count) == 0:
        return np.nan
    cumulative_frequency = np.cumsum(frequency)
    total = cumulative_frequency[-1]
    high_total = cutoff * total
    high_i = np.searchsorted(cumulative_frequency, high_total)
    high_count = count[high_i]
    low_count = np.sqrt(high_count)
    low_i = np.searchsorted(count, low_count)
    if low_i == high_i:
        return np.nan
    return float(np.average(count[low_i:high_i], weights=frequency[low_i:high_i]))


def filter_signature(
    signature,
    min_count: int = 2,
    cutoff: float = CUTOFF,
    low: float = 0.1,
    high: float = 5.0,
):
    coverage = estimate_coverage(signature, min_count=min_count, cutoff=cutoff)
    if np.isnan(coverage):
        raise ValueError("unable to estimate coverage")
    low *= coverage
    high *= coverage
    signature["mins"] = [
        value
        for value, abundance in zip(signature["mins"], signature["abundances"])
        if abundance >= low and abundance < high
    ]
    signature.pop("abundances", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("signature")
    args = parser.parse_args()
    print(args)
    with pam_io.get_input_handle(args.signature) as f:
        signature_files = json.load(f)
    for signature_file in signature_files:
        for signature in signature_file["signatures"]:
            print(len(signature["mins"]))
            filter_signature(signature)
            print(len(signature["mins"]))
