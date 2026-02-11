import argparse
import os
import subprocess
import pandas as pd


class Comparer:
    directory: str
    kmer_size: int
    scaled: float

    def __init__(self, directory: str):
        self.directory = directory
        import warnings

        warnings.warn("Manually set")
        self.kmer_size = 21
        self.scaled = 1000

    @property
    def index(self):
        return os.path.join(self.directory, "clusters.sbt.zip")

    def sketch_reads(self, read_paths: tuple[str], sig: str):
        subprocess.run(
            [
                "sourmash",
                "sketch",
                "--param-string",
                f"k={self.kmer_size},scaled={self.scaled}",
                "--output",
                sig,
                *read_paths,
            ],
            check=True,
        )

    def find_closest(self, query: str, closest: str):
        subprocess.run(
            ["sourmash", "search", query, self.index, "--output", closest], check=True
        )

    def process_reads(self, read_paths: tuple[str], output_directory: str):
        os.makedirs(output_directory, exist_ok=True)
        query = os.path.join(output_directory, "sketch.sig")
        self.sketch_reads(read_paths, query)
        closest = os.path.join(output_directory, "closest.csv")
        self.find_closest(query, closest)
        df = pd.read_csv(closest)
        print(df)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Sketch reads and find closest matches in a sourmash index"
    )
    parser.add_argument(
        "-i",
        "--directory",
        required=True,
        help="Path to the pipeline output directory",
    )
    parser.add_argument(
        "-r",
        "--read",
        action="append",
        required=True,
        help="One or more read files to sketch",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for sketches and results",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    comparer = Comparer(args.directory)
    comparer.process_reads(args.read, args.output_directory)


# import argparse
# import os
# import subprocess
# import pandas as pd
# from dataclasses import dataclass

# DEFAULT_KMER_LENGTH = 25
# DEFAULT_SCALED = 1000


# @dataclass(frozen=True)
# class Comparer:
#     index: str
#     kmer_length: int = DEFAULT_KMER_LENGTH
#     scaled: float = DEFAULT_SCALED

#     def sketch_reads(self, read_paths: tuple[str], sig: str):
#         subprocess.run(
#             [
#                 "sourmash",
#                 "sketch",
#                 "--param-string",
#                 f"k={self.kmer_length},scaled={self.scaled}",
#                 "--output",
#                 sig,
#                 *read_paths,
#             ],
#             check=True,
#         )

#     def find_closest(self, query: str, closest: str):
#         subprocess.run(
#             ["sourmash", "search", query, self.index, "--output", closest], check=True
#         )

#     def process_reads(self, read_paths: tuple[str], output_directory: str):
#         os.makedirs(output_directory, exist_ok=True)
#         query = os.path.join(output_directory, "sketch.sig")
#         self.sketch_reads(read_paths, query)
#         closest = os.path.join(output_directory, "closest.csv")
#         self.find_closest(query, closest)
#         df = pd.read_csv(closest)


# def get_parser():
#     parser = argparse.ArgumentParser(
#         description="Sketch reads and find closest matches in a sourmash index"
#     )
#     parser.add_argument(
#         "-i",
#         "--index",
#         required=True,
#         help="Path to the SBT index or signature collection",
#     )
#     parser.add_argument(
#         "-r",
#         "--reads",
#         required=True,
#         nargs="+",
#         help="One or more read files to sketch",
#     )
#     parser.add_argument(
#         "-o",
#         "--output",
#         required=True,
#         help="Output directory for sketches and results",
#     )
#     parser.add_argument(
#         "-k", "--kmer-length", type=int, default=DEFAULT_KMER_LENGTH, help="k-mer size"
#     )
#     parser.add_argument(
#         "-s",
#         "--scaled",
#         type=int,
#         default=DEFAULT_SCALED,
#         help="scaled factor for sketching",
#     )
#     return parser


# def main():
#     parser = get_parser()
#     args = parser.parse_args()
#     comparer = Comparer(args.index, kmer_length=args.kmer_length, scaled=args.scaled)
#     comparer.process_reads(read_paths, output_directory)
