from . import jaccard, kmc, pam_io, sketch

import argparse
import h5py
import traceback
import multiprocessing
import numpy as np
import os
import sys
import typing
from tempfile import TemporaryDirectory
from tqdm import tqdm as make_progressbar


def main():
    parser = argparse.ArgumentParser(
        description="Generate kmer sketches from a manifest of read sets"
    )
    parser.add_argument("reference_sketch", help="Reference sketch")
    parser.add_argument("query_sketch", help="Query sketch")
    parser.add_argument(
        "--output_phylip",
        "-o",
        default="-",
        help="Output PHYLIP file (defaults to stdout)",
    )
    parser.add_argument(
        "-m",
        "--max_memory",
        type=float,
        default=kmc.MINIMUM_MAX_MEMORY,
        help=f"Max amount of RAM in GB (default {kmc.MINIMUM_MAX_MEMORY}, "
        f">={kmc.MINIMUM_MAX_MEMORY})",
    )
    parser.add_argument(
        "-t",
        "--num_threads",
        type=int,
        default=kmc.NUM_CPUS,
        help=f"Number of threads (default {kmc.NUM_CPUS})",
    )
    args = parser.parse_args()

    if args.num_threads != 1:
        raise NotImplementedError
    reference_names, reference_sketches = sketch.load_sketches(args.reference_sketch)
    query_names, query_sketches = sketch.load_sketches(args.query_sketch)
    distances = jaccard.get_jaccard_distances(reference_sketches, query_sketches)
    with pam_io.get_output_handle(args.output_phylip) as f:
        pam_io.write_distance_matrix(f, names, distances)


if __name__ == "__main__":
    main()
