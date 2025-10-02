import argparse
import numpy as np
import typing
import umap
import warnings
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from . import pam_io


def normalize_embedding(
    z,
) -> np.ndarray[tuple[int, typing.Literal[2]], np.dtype[np.float64]]:
    z = np.asarray(z, dtype=np.float64)
    pca = PCA(n_components=2).fit(z)
    scale = np.sqrt(pca.explained_variance_.sum())
    if scale < 1e-5:
        warnings.warn("Variance is zero; only centering embedding")
        return z - pca.mean_
    matrix = pca.components_.T / scale
    return (z - pca.mean_) @ matrix


def embed(distances, normalize: bool = True, num_jobs: int = 1):
    distances = np.asarray(distances, dtype=np.float64)
    reducer = umap.UMAP(metric="precomputed", n_jobs=num_jobs)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="using precomputed metric; inverse_transform will be unavailable",
            category=UserWarning,
            module="umap.umap_",
        )
        z = reducer.fit_transform(distances)
    if normalize:
        z = normalize_embedding(z)
    return typing.cast(
        np.ndarray[tuple[int, typing.Literal[2]], np.dtype[np.float64]], z
    )  # need cast or Pylance complains


def cluster_embedding(z, eps: float = 0.05, min_samples: int = 10, num_jobs: int = 1):
    clusters = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=num_jobs)
    return clusters.fit(z)


def embed_distances(
    input_phylip: str, num_jobs: int = 1, eps: float = 0.05, min_samples: int = 10
):
    names, distances = pam_io.load_distance_matrix(input_phylip)
    z = embed(distances, num_jobs=num_jobs)
    clusters = cluster_embedding(z, eps=eps, min_samples=min_samples, num_jobs=num_jobs)
    return (names, z, clusters)


def write_embedding(filename: str, names: list[str], z, clusters: DBSCAN):
    with pam_io.get_output_handle(filename) as f:
        print("name", "x", "y", "label", sep="\t", file=f)
        for name, (x, y), label in zip(names, z, clusters.labels_):
            print(name, x, y, label, sep="\t", file=f)


def main():
    parser = argparse.ArgumentParser(
        description="Embed PHYLIP distance matrix using UMAP and cluster with DBSCAN."
    )
    parser.add_argument(
        "input_phylip",
        nargs="?",
        default="-",
        help="Input PHYLIP file (defaults to stdin)",
    )
    parser.add_argument(
        "--output_tsv", "-o", default="-", help="Output TSV file (defaults to stdout)"
    )
    parser.add_argument(
        "--num_jobs",
        "-t",
        type=int,
        default=1,
        help="Number of jobs for embedding and clustering (default: 1)",
    )
    dbscan_group = parser.add_argument_group("DBSCAN clustering options")
    dbscan_group.add_argument(
        "--eps",
        "-e",
        type=float,
        default=0.05,
        help="Maximum distance between neighbouring samples in cluster (default: 0.05)",
    )
    dbscan_group.add_argument(
        "--min_samples",
        "-m",
        type=int,
        default=10,
        help="Minimum samples per cluster (default: 10)",
    )
    args = parser.parse_args()

    names, z, clusters = embed_distances(
        args.input_phylip,
        num_jobs=args.num_jobs,
        eps=args.eps,
        min_samples=args.min_samples,
    )
    write_embedding(args.output_tsv, names, z, clusters)


if __name__ == "__main__":
    main()
