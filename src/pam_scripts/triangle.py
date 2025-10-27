import argparse
from . import jaccard, kmc, pam_io, sketch


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
        "--num_threads",
        "-t",
        type=int,
        default=kmc.NUM_CPUS,
        help=f"Number of threads (default {kmc.NUM_CPUS})",
    )
    args = parser.parse_args()

    if args.num_threads != 1:
        raise NotImplementedError
    names, sketches = sketch.load_sketches(args.input_sketch)
    distances = jaccard.get_pairwise_jaccard_distances(sketches)
    with pam_io.get_output_handle(args.output_phylip) as f:
        pam_io.write_distance_matrix(f, names, distances)


if __name__ == "__main__":
    main()
