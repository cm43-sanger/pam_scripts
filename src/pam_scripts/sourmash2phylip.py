import argparse
import dataclasses
import os
import subprocess
import tempfile
import typing
import zipfile
import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm
from contextlib import contextmanager
from pam_scripts import pam_io
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@contextmanager
def intermediate_filename(filename: typing.Optional[str] = None):
    if filename is None:
        with tempfile.NamedTemporaryFile() as file:
            yield file.name
    else:
        yield filename


@dataclasses.dataclass(frozen=True)
class CondensedDistanceMatrix:
    names: list[str]
    d: npt.NDArray

    def __len__(self):
        return len(self.names)

    @classmethod
    def from_phylip(cls, file: typing.TextIO):
        try:
            num = int(next(file).strip())
            names = []
            d = np.empty(num * (num - 1) // 2)
            start = 0
            for row in range(num):
                line = next(file).strip()
                data = line.split(maxsplit=row + 1)
                names.append(data[0])
                stop = start + row
                d[start:stop] = np.array(data[1 : row + 1], dtype=float)
                start = stop
        except ValueError as e:
            raise ValueError("invalid phylip file") from e
        return cls(names=names, d=d)

    @classmethod
    def from_sourmash_branchwater(
        cls,
        zip_path: str,
        kmer_length: int,
        csv_filename: typing.Optional[str] = None,
        chunksize=8_192,
    ):
        with zipfile.ZipFile(zip_path) as zf, zf.open("SOURMASH-MANIFEST.csv") as fp:
            df = pd.read_csv(fp, comment="#")
        num = len(df)
        total = num * (num - 1) // 2
        names_lookup = {name: i for i, name in enumerate(df["name"])}
        d = np.empty(total)
        # with tempfile.NamedTemporaryFile("w") as csv_file:
        #     subprocess.run(
        #         [
        #             "sourmash",
        #             "scripts",
        #             "pairwise",
        #             zip_path,
        #             f"--output={csv_file.name}",
        #             "--threshold=0.0",
        #             f"--cores={os.cpu_count() or 1}",
        #             "--ani",
        #         ],
        #         check=True,
        #         text=True,
        #     )
        #     with tqdm.tqdm(total=total, desc="Pivoting", unit_scale=True) as pbar:
        #         for chunk in pd.read_csv(csv_file.name, chunksize=chunksize):
        #             i = chunk["query_name"].map(names_lookup).to_numpy(dtype=np.int64)
        #             j = chunk["match_name"].map(names_lookup).to_numpy(dtype=np.int64)
        #             lower = np.minimum(i, j)
        #             upper = np.maximum(i, j)
        #             k = upper * (upper - 1) // 2 + lower
        #             d[k] = 1.0 - chunk["average_containment_ani"].to_numpy()
        #             pbar.update(len(chunk))
        with intermediate_filename(csv_filename) as csv_filename:
            subprocess.run(
                [
                    "sourmash",
                    "scripts",
                    "pairwise",
                    zip_path,
                    f"--output={csv_filename}",
                    "--threshold=0.0",
                    f"--ksize={kmer_length}",
                    f"--cores={os.cpu_count() or 1}",
                    "--ani",
                ],
                check=True,
                text=True,
            )
            import shutil

            shutil.copyfile(
                csv_filename, "/warehouse/hpag_wh01/cm43/scripts/sourmash.csv"
            )
            with tqdm.tqdm(total=total, desc="Pivoting", unit_scale=True) as pbar:
                for chunk in pd.read_csv(csv_filename, chunksize=chunksize):
                    for lineno, name in enumerate(chunk["query_name"], start=1):
                        if name not in names_lookup:
                            print(f"Missing query {name!r} at line {lineno}")
                    for lineno, name in enumerate(chunk["match_name"], start=1):
                        if name not in names_lookup:
                            print(f"Missing match {name!r} at line {lineno}")
                    i = chunk["query_name"].map(names_lookup).to_numpy(dtype=np.int64)
                    j = chunk["match_name"].map(names_lookup).to_numpy(dtype=np.int64)
                    lower = np.minimum(i, j)
                    upper = np.maximum(i, j)
                    k = upper * (upper - 1) // 2 + lower
                    try:
                        d[k] = 1.0 - chunk["average_containment_ani"].to_numpy()
                    except IndexError:
                        chunk.to_csv("/warehouse/hpag_wh01/cm43/scripts/bad.csv")
                        raise RuntimeError
                    pbar.update(len(chunk))
        return cls(names=df["name"].to_list(), d=d)

    def inflate(self):
        return squareform(self.d)

    def to_phylip(self, fp: typing.TextIO):
        num = len(self.names)
        length = max(map(len, self.names))
        print(num, file=fp)
        start = 0
        for row, name in enumerate(self.names):
            stop = start + row
            print(
                name.ljust(length),
                *(format(value, ".6f") for value in self.d[start:stop]),
                file=fp,
            )
            start = stop

    def cluster(self, threshold: float):
        Z = linkage(1.0 - self.d, method="complete")
        labels = fcluster(Z, threshold, criterion="distance")
        clusters = {}
        indices = []
        new_names = []
        for i, label in enumerate(labels):
            if label in clusters:
                continue
            clusters[label] = i
            indices.append(i)
            new_names.append(self.names[i])
        indices = np.array(indices)
        ani = squareform(self.d)
        new_ani = ani[np.ix_(indices, indices)]
        return (
            labels,
            CondensedDistanceMatrix(names=new_names, d=squareform(new_ani)),
        )


# def get_d(zip_path: str, chunksize=8_192):
#     with zipfile.ZipFile(zip_path) as zf, zf.open("SOURMASH-MANIFEST.csv") as fp:
#         df = pd.read_csv(fp, comment="#")
#     num = len(df)
#     total = num * (num - 1) // 2
#     names_lookup = {name: i for i, name in enumerate(df["name"])}
#     ani = np.empty(total)
#     with tempfile.NamedTemporaryFile("w") as csv_file:
#         subprocess.run(
#             [
#                 "sourmash",
#                 "scripts",
#                 "pairwise",
#                 zip_path,
#                 f"--output={csv_file.name}",
#                 "--threshold=0.0",
#                 f"--cores={os.cpu_count() or 1}",
#                 "--ani",
#             ],
#             check=True,
#             text=True,
#         )
#         with tqdm.tqdm(total=total, desc="Pivoting", unit_scale=True) as pbar:
#             for chunk in pd.read_csv(csv_file.name, chunksize=chunksize):
#                 i = chunk["query_name"].map(names_lookup).to_numpy(dtype=np.int64)
#                 j = chunk["match_name"].map(names_lookup).to_numpy(dtype=np.int64)
#                 lower = np.minimum(i, j)
#                 upper = np.maximum(i, j)
#                 k = upper * (upper - 1) // 2 + lower
#                 ani[k] = chunk["average_containment_ani"].to_numpy()
#                 pbar.update(len(chunk))
#     print(
#         ani.min(),
#         ani.max(),
#         ani.mean(),
#         ani.std(),
#     )
#     return CondensedDistanceMatrix(names=df["name"].to_list(), d=1.0 - ani)


def get_d(zip_path: str, chunksize=8_192):
    with zipfile.ZipFile(zip_path) as zf, zf.open("SOURMASH-MANIFEST.csv") as fp:
        df = pd.read_csv(fp, comment="#")
    num = len(df)
    total = num * (num - 1) // 2
    names_lookup = {name: i for i, name in enumerate(df["name"])}
    d = np.empty(total)
    with tempfile.NamedTemporaryFile("w") as csv_file:
        subprocess.run(
            [
                "sourmash",
                "scripts",
                "pairwise",
                zip_path,
                f"--output={csv_file.name}",
                "--threshold=0.0",
                f"--cores={os.cpu_count() or 1}",
                "--ani",
            ],
            check=True,
            text=True,
        )
        with tqdm.tqdm(total=total, desc="Assembling", unit_scale=True) as pbar:
            for chunk in pd.read_csv(csv_file.name, chunksize=chunksize):
                i = chunk["query_name"].map(names_lookup).to_numpy(dtype=np.int64)
                j = chunk["match_name"].map(names_lookup).to_numpy(dtype=np.int64)
                lower = np.minimum(i, j)
                upper = np.maximum(i, j)
                k = upper * (upper - 1) // 2 + lower
                d[k] = 1.0 - chunk["average_containment_ani"].to_numpy()
                pbar.update(len(chunk))
    print(
        d.min(),
        d.max(),
        d.mean(),
        d.std(),
    )
    return CondensedDistanceMatrix(names=df["name"].to_list(), d=1.0 - d)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate ANI distances for sourmash branchwater sketch zip."
    )
    parser.add_argument("zip", help="Zip file path")
    parser.add_argument(
        "-k", "--kmer_length", type=int, required=True, help="Kmer length"
    )
    parser.add_argument(
        "-o", "--output", default="-", help="Output phylip file (defaults to stdin)"
    )
    args = parser.parse_args()
    matrix = CondensedDistanceMatrix.from_sourmash_branchwater(
        args.zip, kmer_length=args.kmer_length
    )
    with pam_io.get_output_handle(args.output) as f:
        matrix.to_phylip(f)


# if __name__ == "__main__":
#     matrix = CondensedDistanceMatrix.from_sourmash_branchwater(
#         "/Users/cm43/Documents/scripts/test.zip",
#         csv_filename="/Users/cm43/Documents/scripts/test.csv",
#     )
#     with open("sourmash2phylip.phylip", "w") as fp:
#         matrix.to_phylip(fp)

#     labels, new_ani = matrix.cluster(threshold=0.005)

#     print(len(matrix), len(new_ani))

#     import matplotlib.pyplot as plt

#     fig, axis = plt.subplots()
#     axis.hist(matrix.d, bins=100)
#     fig.tight_layout()
#     plt.show()
