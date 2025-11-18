import argparse
from . import _core, jaccard, pam_io, sketch


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
        "-s",
        "--scale",
        type=int,
        default=None,
        help="Downsampling scale factor (default: database value)",
    )
    parser.add_argument(
        "-t",
        "--num_threads",
        type=int,
        default=_core.NUM_CPUS,
        help=f"Number of threads (default {_core.NUM_CPUS})",
    )
    args = parser.parse_args()

    if args.num_threads != 1:
        raise NotImplementedError
    reference_names, reference_sketches = sketch.load_sketches(
        args.reference_sketch, scale=args.scale
    )
    query_names, query_sketches = sketch.load_sketches(
        args.query_sketch, scale=args.scale
    )
    distances = jaccard.get_distances(reference_sketches, query_sketches)
    with pam_io.get_output_handle(args.output_phylip) as f:
        pam_io.write_distance_matrix(f, names, distances)


if __name__ == "__main__":
    main()
