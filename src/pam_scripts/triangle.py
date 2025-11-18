import argparse
from . import _core, jaccard, pam_io, sketch


def main():
    parser = argparse.ArgumentParser(
        description="Calculate pairwise distances in a sketch."
    )
    parser.add_argument("input_sketch", help="Input sketch file")
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
        "--num_threads",
        "-t",
        type=int,
        default=_core.NUM_CPUS,
        help=f"Number of threads (default {_core.NUM_CPUS})",
    )
    args = parser.parse_args()

    if args.num_threads != 1:
        raise NotImplementedError
    names, sketches, kmer_length = sketch.load_sketches(
        args.input_sketch, scale=args.scale
    )
    distances = jaccard.get_pairwise_distances(sketches, kmer_length)
    with pam_io.get_output_handle(args.output_phylip) as f:
        pam_io.write_distance_matrix(f, names, distances)


if __name__ == "__main__":
    main()
